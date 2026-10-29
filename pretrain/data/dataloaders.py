import glob
import os
import numpy as np
import torch

from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset as TorchDataset, Subset

from data.datasamplers import StatefulSequentialSampler, StatefulRandomSampler, StatefulDistributedSampler


class FinewebShardBinDataset(TorchDataset):
  """Read KellerJordan/nanoGPT FineWeb .bin shards with 256-int32 header + uint16 tokens.

  Each sample is a non-overlapping chunk of length seq_len+1 from within one shard.
  """

  HEADER_INTS = 256

  def __init__(self, pattern_or_dir, seq_len, split='train'):
    self.seq_len = seq_len
    self.sample_len = seq_len + 1
    if os.path.isdir(pattern_or_dir):
      pat = os.path.join(pattern_or_dir, f'fineweb_{split}_*.bin')
      self.files = sorted(glob.glob(pat))
    else:
      self.files = sorted(glob.glob(pattern_or_dir))
    if not self.files:
      raise ValueError(f'No shard files found for {pattern_or_dir} split={split}')

    self.shards = []
    self.cum_samples = []
    total = 0
    for path in self.files:
      hdr = np.fromfile(path, dtype=np.int32, count=self.HEADER_INTS)
      if len(hdr) < 3 or hdr[0] != 20240520:
        raise ValueError(f'Invalid fineweb bin header: {path}')
      ntok = int(hdr[2])
      data = np.memmap(path, mode='r', dtype=np.uint16, offset=self.HEADER_INTS * 4, shape=(ntok,))
      nsamp = ntok // self.sample_len
      if nsamp == 0:
        continue
      self.shards.append((path, data, nsamp))
      total += nsamp
      self.cum_samples.append(total)
    if not self.shards:
      raise ValueError('No usable shards with at least one sample')
    self.n = total

  def __len__(self):
    return self.n

  def __getitem__(self, idx):
    shard_idx = 0
    while idx >= self.cum_samples[shard_idx]:
      shard_idx += 1
    prev = 0 if shard_idx == 0 else self.cum_samples[shard_idx - 1]
    local_idx = idx - prev
    _, data, _ = self.shards[shard_idx]
    start = local_idx * self.sample_len
    arr = np.asarray(data[start : start + self.sample_len], dtype=np.int64)
    return {'input_ids': torch.from_numpy(arr)}


def get_dataloaders(cfg):
  """Load trainset and perhaps validset. Returns correspondent DataLoaders."""

  use_kj_bin = getattr(cfg, 'trainset_bin_dir', None) is not None

  if use_kj_bin:
    train_set = FinewebShardBinDataset(cfg.trainset_bin_dir, cfg.seq_len, split='train')
  else:
    train_set = load_from_disk(cfg.trainset_path)
    if not isinstance(train_set, Dataset):
      raise ValueError('dataset should be a datasets.Dataset')

  train_sampler = _get_sampler(train_set, cfg)

  def collate_fn(batch):
    return {
      'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0),
      'docs_lengths': [x['docs_lengths'].tolist() for x in batch],
    }

  has_docs_lengths = (not use_kj_bin) and ('docs_lengths' in train_set.column_names)

  trainloader = DataLoader(
    train_set,
    sampler=train_sampler,
    batch_size=cfg.micro_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=True if cfg.num_workers > 0 else False,
    collate_fn=collate_fn if has_docs_lengths else None,
  )

  if not cfg.validset_path and getattr(cfg, 'validset_bin_dir', None) is None:
    validloader = None
  else:
    if getattr(cfg, 'validset_bin_dir', None) is not None:
      valid_set = FinewebShardBinDataset(cfg.validset_bin_dir, cfg.seq_len, split='val')
    else:
      valid_set = load_from_disk(cfg.validset_path)
      if not isinstance(valid_set, Dataset):
        raise ValueError("'dataset' should be a datasets.Dataset")

      if getattr(cfg, 'valid_tokens', False):
        valid_rows = cfg.valid_tokens // (cfg.seq_len + 1)
        valid_set = valid_set.take(valid_rows)

    if dist.is_initialized() and getattr(cfg, 'eval_rank0_only', False):
      if dist.get_rank() != 0:
        valid_set = Subset(valid_set, [])
      valid_sampler = SequentialSampler(valid_set)
    elif dist.is_initialized():
      valid_sampler = DistributedSampler(valid_set, drop_last=True)
    else:
      valid_sampler = SequentialSampler(valid_set)

    validloader = DataLoader(
      valid_set,
      batch_size=cfg.micro_batch_size,
      drop_last=True,
      shuffle=False,
      sampler=valid_sampler,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=False,
      collate_fn=collate_fn if has_docs_lengths else None,
    )

  return trainloader, validloader


def _get_sampler(train_set, cfg):
  ddp = dist.is_initialized()

  if cfg.sampler == 'random':
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.sampler_seed, drop_last=True)
    else:
      sampler = RandomSampler(
        train_set, generator=torch.Generator().manual_seed(cfg.sampler_seed) if cfg.sampler_seed else None
      )

  elif cfg.sampler == 'sequential':
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=False, drop_last=True)
    else:
      sampler = SequentialSampler(train_set)

  elif cfg.sampler == 'stateful_random':
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      sampler = StatefulDistributedSampler(
        train_set, batch_size=cfg.micro_batch_size, seed=cfg.sampler_seed, start_iter=micro_step_start
      )
    else:
      sampler = StatefulRandomSampler(
        train_set, batch_size=cfg.micro_batch_size, shuffle=True, seed=cfg.sampler_seed, start_idx=micro_step_start
      )

  elif cfg.sampler == 'stateful_sequential':
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      raise NotImplementedError('StatefulDistributedSampler currently needs a seed.')
    else:
      sampler = StatefulSequentialSampler(train_set, batch_size=cfg.micro_batch_size, start_idx=micro_step_start)

  else:
    raise NotImplementedError(f'Sampler {cfg.sampler} is not implemented.')

  return sampler
