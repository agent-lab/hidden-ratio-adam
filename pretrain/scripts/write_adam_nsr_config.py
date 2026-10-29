"""Write a local plainLM config for an already prepared Hugging Face dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--trainset-path', required=True, help='Path saved by datasets.Dataset.save_to_disk.')
  parser.add_argument('--validset-path', default=None, help='Optional validation dataset path.')
  parser.add_argument('--out', default='config/local_adam_nsr.yaml', help='Output YAML path.')
  parser.add_argument('--seq-len', type=int, default=2048)
  parser.add_argument('--vocab-size', type=int, default=50280)
  parser.add_argument('--steps-budget', type=int, default=4800)
  parser.add_argument('--micro-batch-size', type=int, default=32)
  parser.add_argument('--grad-accumulation-steps', type=int, default=8)
  parser.add_argument('--lr', type=float, default=3e-3)
  parser.add_argument('--weight-decay', type=float, default=0.1)
  parser.add_argument('--beta', type=float, default=0.90, help='Adam-NSR uses beta1 == beta2.')
  parser.add_argument('--out-dir', default='./outputs')
  parser.add_argument('--exp-name', default='adam_nsr_run')
  parser.add_argument('--wandb-project', default='plainlm')
  parser.add_argument('--wandb-entity', default=None)
  parser.add_argument('--use-wandb', action='store_true')
  return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
  return {
    'deterministic': False,
    'seed': 100,
    'trainset_path': args.trainset_path,
    'vocab_size': args.vocab_size,
    'seq_len': args.seq_len,
    'sampler': 'sequential',
    'sampler_seed': None,
    'num_workers': 4,
    'eval': args.validset_path is not None,
    'validset_path': args.validset_path,
    'eval_every_steps': 200 if args.validset_path is not None else None,
    'model': 'transformer',
    'd_model': 768,
    'mlp_class': 'glu',
    'expand': '8/3',
    'n_layers': 12,
    'n_heads': 12,
    'rms_norm': True,
    'tie_embeddings': False,
    'qk_norm': True,
    'embed_norm': True,
    'torch_compile': True,
    'steps_budget': args.steps_budget,
    'micro_batch_size': args.micro_batch_size,
    'grad_accumulation_steps': args.grad_accumulation_steps,
    'dtype': 'bfloat16',
    'optim': 'adam_nsr',
    'fused_optim': False,
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'beta1': args.beta,
    'beta2': args.beta,
    'eps': 1e-8,
    'grad_clip': 1.0,
    'y_max': 64.0,
    'moment1_dtype': 'fp32',
    'moment1_quant': 'per_column',
    'moment2_dtype': 'fp32',
    'pseudo_quantize_state_dtype': None,
    'nsr_compute_dtype': 'fp32',
    'skip_update_on_bad_y': False,
    'scheduler': 'warmup_cosine',
    'warmup_steps': 0.1,
    'cooldown_steps': None,
    'lr_start': 0.0,
    'lr_end': 1e-5,
    'lr_end_pct': None,
    'log_every_steps': 1,
    'print_progress': True,
    'use_wandb': args.use_wandb,
    'wandb_entity': args.wandb_entity,
    'wandb_project': args.wandb_project,
    'wandb_dir': './wandb',
    'wandb_run_name': args.exp_name,
    'exp_name': args.exp_name,
    'out_dir': args.out_dir,
    'over_write': True,
    'resume': False,
    'resume_step': None,
    'resume_exp_name': None,
    'save_last_checkpoint': False,
    'save_intermediate_checkpoints': False,
    'save_every_steps': None,
  }


def main() -> None:
  args = parse_args()
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open('w', encoding='utf-8') as handle:
    yaml.safe_dump(build_config(args), handle, sort_keys=False)
  print(f'Wrote {out_path}')


if __name__ == '__main__':
  main()
