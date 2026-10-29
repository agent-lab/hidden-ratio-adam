#!/usr/bin/env python
"""Download cached KellerJordan FineWeb GPT-2 .bin shards from Hugging Face.

These shards are already tokenized with the GPT-2 tokenizer and use the same
256-int32 header + uint16 payload format consumed by plainLM's bin dataloader.
Use this path when training GPT-2-vocab models such as the default transformer
or Pythia-style models. For the custom llama-20M TinyLlama-tokenizer data, use
`scripts/build_fineweb_bins.py` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


PRESETS = {
  'fineweb10b': {
    'repo_id': 'kjj0/fineweb10B-gpt2',
    'local_dir': 'fineweb10B',
    'prefix': 'fineweb',
    'num_chunks': 103,
  },
  'fineweb100b': {
    'repo_id': 'kjj0/fineweb100B-gpt2',
    'local_dir': 'fineweb100B',
    'prefix': 'fineweb',
    'num_chunks': 1030,
  },
  'finewebedu10b': {
    'repo_id': 'kjj0/finewebedu10B-gpt2',
    'local_dir': 'finewebedu10B',
    'prefix': 'finewebedu',
    'num_chunks': 99,
  },
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--preset',
    choices=sorted(PRESETS),
    default='fineweb10b',
    help='Which cached GPT-2-tokenized dataset to download.',
  )
  parser.add_argument(
    '--output-dir',
    type=Path,
    default=None,
    help='Directory for downloaded .bin shards. Defaults to ./data/<preset-dir>.',
  )
  parser.add_argument(
    '--num-train-chunks',
    type=int,
    default=None,
    help='Number of train shards to download. Defaults to the full preset.',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  preset = PRESETS[args.preset]
  output_dir = args.output_dir or Path('./data') / preset['local_dir']
  output_dir.mkdir(parents=True, exist_ok=True)
  num_chunks = preset['num_chunks'] if args.num_train_chunks is None else args.num_train_chunks

  def get(filename: str) -> None:
    target = output_dir / filename
    if target.exists():
      print(f'[skip] {target}')
      return
    print(f'[download] {filename}')
    hf_hub_download(
      repo_id=preset['repo_id'],
      filename=filename,
      repo_type='dataset',
      local_dir=output_dir,
    )

  prefix = preset['prefix']
  get(f'{prefix}_val_000000.bin')
  for i in range(1, num_chunks + 1):
    get(f'{prefix}_train_{i:06d}.bin')
  print(f'[done] downloaded {args.preset} shards to {output_dir}')


if __name__ == '__main__':
  main()
