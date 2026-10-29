import json
import math
import os
import time
from collections.abc import Iterable

import torch


_FP8_QMAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}


def _paper_y_table():
    """Return the fixed 16-value codebook used by the paper experiments."""
    return (
        0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
        6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
    )


def _normalize_y_table(y_table):
    if isinstance(y_table, str) and y_table == "paper":
        return _paper_y_table()
    if y_table is None:
        return None
    if torch.is_tensor(y_table):
        y_table = y_table.detach().cpu().tolist()
    elif not isinstance(y_table, (list, tuple)) and isinstance(y_table, Iterable) and not isinstance(y_table, (str, bytes)):
        # Hydra/OmegaConf passes list CLI overrides as ListConfig, which behaves
        # like an iterable but is not a built-in list/tuple.
        y_table = list(y_table)
    if not isinstance(y_table, (list, tuple)):
        raise ValueError("y_table must be 'paper', None, or a list-like sequence of finite floats")
    if len(y_table) == 0:
        raise ValueError("y_table must contain at least one entry")

    values = tuple(float(v) for v in y_table)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("y_table entries must be finite")
    if any(v < 0.0 for v in values):
        raise ValueError("y_table entries must be non-negative")
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        raise ValueError("y_table must be strictly increasing")
    return values


def _resolve_state_spec(name):
    if name is None:
        return None
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower().strip()
    if key == "int4":
        return "int4"
    if key == "small":
        return "small"
    if key in ("smallv1", "small_legacy64"):
        return "smallv1"
    if key == "smallv2":
        return "smallv2"
    if key == "big":
        return "big"
    if key == "yfp4":
        return "yfp4"
    if key in ("yfp4tab", "yfp4_tab"):
        return "yfp4_tab"
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "float8_e4m3": torch.float8_e4m3fn,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "float8_e5m2": torch.float8_e5m2,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _resolve_compute_dtype(name):
    if name is None:
        return None
    if isinstance(name, torch.dtype):
        if name not in (torch.float32, torch.float64):
            raise ValueError(f"Unsupported compute dtype: {name}")
        return name
    key = str(name).lower().strip()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported compute dtype: {name}")
    return mapping[key]


def _fake_quantize_int4_per_group(x_fp32, group_mode="per_column", eps=1e-12):
    if x_fp32.ndim == 1:
        group_mode = "per_tensor"

    qmin, qmax = -8.0, 7.0

    if group_mode == "per_tensor":
        amax = x_fp32.abs().max().clamp_min(eps)
        scale = amax / qmax
    elif group_mode == "per_column":
        if x_fp32.ndim != 2:
            amax = x_fp32.abs().max().clamp_min(eps)
            scale = amax / qmax
        else:
            amax = x_fp32.abs().amax(dim=0, keepdim=True).clamp_min(eps)
            scale = amax / qmax
    elif group_mode == "per_channel":
        if x_fp32.ndim == 1:
            amax = x_fp32.abs().max().clamp_min(eps)
            scale = amax / qmax
        else:
            reduce_dims = tuple(range(1, x_fp32.ndim))
            amax = x_fp32.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
            scale = amax / qmax
    else:
        raise ValueError(f"Unknown group_mode: {group_mode}")

    x_int = torch.round(x_fp32 / scale).clamp(qmin, qmax)
    return x_int * scale


def _fake_quantize_state(x_fp32, q_spec, group_mode="per_column", eps=1e-12):
    if q_spec is None or q_spec == torch.float32:
        return x_fp32
    if q_spec == "int4":
        return _fake_quantize_int4_per_group(x_fp32, group_mode=group_mode, eps=eps)
    if q_spec in ("small", "smallv1", "smallv2", "big", "yfp4", "yfp4_tab"):
        return x_fp32
    if q_spec in _FP8_QMAX:
        amax = x_fp32.abs().max().clamp_min(eps)
        scale = _FP8_QMAX[q_spec] / amax
        return (x_fp32 * scale).to(q_spec).float() / scale
    return x_fp32.to(q_spec).float()


def _get_y_table_key(q_spec):
    if q_spec == "small":
        return "small"
    if q_spec == "smallv1":
        return "smallv1"
    if q_spec == "smallv2":
        return "smallv2"
    if q_spec == "big":
        return "big"
    if q_spec in ("yfp4", "yfp4_tab"):
        return q_spec
    return "default"


def _get_y_table_values(q_spec, explicit_y_table=None):
    if explicit_y_table is not None:
        return explicit_y_table
    small_legacy64 = [
        0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
        4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0,
    ]

    small = [
        0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5,
        2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0
    ]

    big = [
        0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
        6.0, 8.0, 12.0, 16.0, 24.0, 48.0, 64.0, 96.0
    ]

    tables = {
        "default": small,
        "small": small,
        "smallv1": small_legacy64,
        "small_legacy64": small_legacy64,
        "big": big,
        "smallv2": small,
        "yfp4": small,
        "yfp4_tab": big,
    }
    return tables[_get_y_table_key(q_spec)]


def _get_y_table_cache_key(q_spec, explicit_y_table=None):
    if explicit_y_table is not None:
        return ("explicit", explicit_y_table)
    return ("preset", _get_y_table_key(q_spec))


def _uses_y_table(q_spec, explicit_y_table=None):
    return explicit_y_table is not None or q_spec in ("small", "smallv1", "smallv2", "big", "yfp4", "yfp4_tab")


def _format_table_value(v):
    text = f"{float(v):g}"
    return text.replace("-", "m").replace(".", "p")


class AdamNSR(torch.optim.Optimizer):
    """plainLM-compatible NSR optimizer.

    Core NSR state/update:
    - maintain exp_avg = m_t
    - maintain y_t via ratio recurrence
    - quantize y_t with unsigned table-FP4 lookup

    Compatibility extension for plainLM:
    - support moment1_dtype / moment2_dtype / pseudo_quantize_state_dtype
    - stored state can be cast/quantized, then restored to FP32 for math

    Notes:
    - This implementation requires beta1 == beta2, following the referenced
      NSR variant.
    - `moment1_dtype` applies to exp_avg / m_t storage.
      `int4` is supported with `moment1_quant` controlling granularity.
    - `moment2_dtype` applies to y_t storage before restoring to FP32.
      With `y_table=None`, `small`/`big` select legacy lookup tables,
      while non-table variants clip by `y_max` without codebook quantization.
      `yfp4`/`yfp4tab` are kept as backward-compatible aliases.
    - `y_table="paper"` uses the internal paper codebook by default.
      An explicit monotonic sequence overrides it; the effective clamp is
      the largest codebook value. `None` preserves legacy non-table/preset modes.
    - `compute_dtype` controls the transient math dtype used for the NSR
      recurrence and update computation. Storage behavior is unchanged.
      Supported values are `fp32` and `fp64`/`float64`.
    - `skip_update_on_bad_y=False` is the default behavior: if `y_raw`
      is invalid or exceeds the clip range, state is clamped but the parameter
      still updates using the clamped value. Set it to `True` to zero the
      corresponding parameter update for that step.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.92, 0.92),
        eps=1e-8,
        weight_decay=0.0,
        y_max=400.0,
        y_table="paper",
        moment1_dtype=None,
        moment1_quant="per_column",
        moment2_dtype=None,
        pseudo_quantize_state_dtype=None,
        compute_dtype=None,
        skip_update_on_bad_y=False,
        transformed_y_hist_log_dir=None,
        transformed_y_hist_log_every=0,
        transformed_y_hist_bins=25,
        transformed_y_hist_min=0.0,
        transformed_y_hist_max=20.0,
        transformed_y_hist_scale="beta_over_one_minus_beta",
        transformed_y_hist_run_name=None,
        transformed_y_hist_total_steps=None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if betas[0] != betas[1]:
            raise ValueError(f"AdamNSR requires equal betas, got betas={betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        explicit_y_table = _normalize_y_table(y_table)
        effective_y_max = float(explicit_y_table[-1]) if explicit_y_table is not None else float(y_max)
        shared_dtype = _resolve_state_spec(pseudo_quantize_state_dtype)
        m1_spec = _resolve_state_spec(moment1_dtype) if moment1_dtype is not None else shared_dtype
        if m1_spec is None and moment1_quant is not None:
            m1_spec = "int4"
        if m1_spec == "int4" and moment1_quant is None:
            moment1_quant = "per_column"
        y2_spec = _resolve_state_spec(moment2_dtype) if moment2_dtype is not None else shared_dtype
        compute_spec = _resolve_compute_dtype(compute_dtype)
        hist_log_every = int(transformed_y_hist_log_every or 0)
        hist_bins = int(transformed_y_hist_bins)
        hist_min = float(transformed_y_hist_min)
        hist_max = float(transformed_y_hist_max)
        if hist_log_every < 0:
            raise ValueError("transformed_y_hist_log_every must be non-negative")
        if hist_bins <= 0:
            raise ValueError("transformed_y_hist_bins must be positive")
        if hist_max <= hist_min:
            raise ValueError("transformed_y_hist_max must be greater than transformed_y_hist_min")
        if transformed_y_hist_scale not in (
            "stored_y",
            "one_minus_beta_over_beta",
            "beta_over_one_minus_beta",
        ):
            raise ValueError(
                "transformed_y_hist_scale must be one of: stored_y, "
                "one_minus_beta_over_beta, beta_over_one_minus_beta"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            y_max=effective_y_max,
            y_table=explicit_y_table,
            moment1_dtype=m1_spec,
            moment1_quant=moment1_quant,
            moment2_dtype=y2_spec,
            compute_dtype=compute_spec,
            skip_update_on_bad_y=bool(skip_update_on_bad_y),
        )
        super().__init__(params, defaults)
        self._tables = {}
        self._table_hit_window = 100
        self._table_hit_accum = None
        self._table_hit_total = 0
        self._table_hit_steps = 0
        self._pending_log_metrics = {}
        self._transformed_y_hist_log_dir = transformed_y_hist_log_dir
        self._transformed_y_hist_log_every = hist_log_every
        self._transformed_y_hist_bins = hist_bins
        self._transformed_y_hist_min = hist_min
        self._transformed_y_hist_max = hist_max
        self._transformed_y_hist_scale = transformed_y_hist_scale
        self._transformed_y_hist_run_name = transformed_y_hist_run_name
        self._transformed_y_hist_total_steps = (
            None if transformed_y_hist_total_steps is None else int(transformed_y_hist_total_steps)
        )
        self._transformed_y_hist_edges = torch.linspace(hist_min, hist_max, hist_bins + 1, dtype=torch.float64)

    def _table_tensor(self, ref, q_spec, explicit_y_table=None):
        table_key = _get_y_table_cache_key(q_spec, explicit_y_table)
        cached = self._tables.get(table_key)
        if cached is None or cached.device != ref.device or cached.dtype != torch.float32:
            cached = torch.tensor(_get_y_table_values(q_spec, explicit_y_table), device=ref.device, dtype=torch.float32)
            self._tables[table_key] = cached
        return cached

    def _quantize_y_table(
        self,
        y_work,
        y_clip_max,
        q_spec,
        explicit_y_table=None,
        return_idx=False,
        return_counts=False,
        chunk_size=4_000_000,
    ):
        try:
            from torch.distributed.tensor import DTensor
        except Exception:
            DTensor = None

        if return_idx and return_counts:
            raise ValueError("Only one of return_idx and return_counts can be True")

        if DTensor is not None and isinstance(y_work, DTensor):
            result = self._quantize_y_table(
                y_work.to_local(),
                y_clip_max,
                q_spec,
                explicit_y_table=explicit_y_table,
                return_idx=return_idx,
                return_counts=return_counts,
                chunk_size=chunk_size,
            )
            if return_idx or return_counts:
                quantized_local, aux = result
            else:
                quantized_local = result
            quantized = DTensor.from_local(
                quantized_local,
                device_mesh=y_work.device_mesh,
                placements=y_work.placements,
                run_check=False,
            )
            if return_idx:
                return quantized, aux
            if return_counts:
                return quantized, aux
            return quantized

        table = self._table_tensor(y_work, q_spec, explicit_y_table)
        if table.dtype != y_work.dtype:
            table = table.to(dtype=y_work.dtype)
        y_work = torch.nan_to_num(y_work, nan=y_clip_max, posinf=y_clip_max, neginf=0.0)
        y_work = y_work.clamp(min=0.0, max=min(float(y_clip_max), float(table[-1].item())))
        flat = y_work.reshape(-1)
        # Quantize by midpoint binning instead of building a full [N, table_len]
        # distance matrix. This matches nearest-neighbor quantization for a sorted
        # table and keeps the tie-break on exact midpoints on the lower entry,
        # matching torch.argmin over abs distances.
        mids = 0.5 * (table[:-1] + table[1:])
        quantized_flat = torch.empty_like(flat)
        idx_flat = torch.empty_like(flat, dtype=torch.long) if return_idx else None
        counts = torch.zeros(len(table), device=flat.device, dtype=torch.float64) if return_counts else None

        for start in range(0, flat.numel(), chunk_size):
            end = min(start + chunk_size, flat.numel())
            chunk = flat[start:end]
            idx_chunk = torch.bucketize(chunk, mids)
            quantized_flat[start:end] = table[idx_chunk]
            if idx_flat is not None:
                idx_flat[start:end] = idx_chunk
            if counts is not None:
                counts.add_(torch.bincount(idx_chunk, minlength=len(table)).to(torch.float64))

        quantized = quantized_flat.reshape_as(y_work)
        if return_idx:
            return quantized, idx_flat
        if return_counts:
            return quantized, counts
        return quantized

    def _update_table_hit_window(self, step, table, step_counts):
        if self._table_hit_accum is None or self._table_hit_accum.numel() != step_counts.numel():
            self._table_hit_accum = torch.zeros_like(step_counts, dtype=torch.float64)
            self._table_hit_total = 0
            self._table_hit_steps = 0

        self._table_hit_accum += step_counts.to(torch.float64)
        self._table_hit_total += int(step_counts.sum().item())
        self._table_hit_steps += 1

        if step % self._table_hit_window != 0:
            return

        denom = max(self._table_hit_total, 1)
        metrics = {
            'nsr/y_table_hit_window_steps': self._table_hit_steps,
            'nsr/y_table_hit_window_total': self._table_hit_total,
            'nsr/y_table_min': float(table[0].item()),
            'nsr/y_table_max': float(table[-1].item()),
            'nsr/y_table_len': int(table.numel()),
        }
        for idx, table_value in enumerate(table.tolist()):
            key = f'nsr/y_table_hit_share_100/code{idx:02d}_{_format_table_value(table_value)}'
            metrics[key] = float(self._table_hit_accum[idx].item()) / float(denom)

        self._pending_log_metrics = metrics
        self._table_hit_accum.zero_()
        self._table_hit_total = 0
        self._table_hit_steps = 0

    def consume_logging_metrics(self):
        metrics = self._pending_log_metrics
        self._pending_log_metrics = {}
        return metrics

    def _transformed_y_hist_factor(self, beta):
        if self._transformed_y_hist_scale in ("stored_y", "one_minus_beta_over_beta"):
            return 1.0
        return (float(beta) / (1.0 - float(beta))) ** 2

    def _transformed_y_hist_output_path(self):
        if not self._transformed_y_hist_log_dir:
            return None
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        run_name = self._transformed_y_hist_run_name or os.environ.get("WANDB_NAME") or "run"
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(run_name))
        return os.path.join(self._transformed_y_hist_log_dir, f"{safe_name}.rank{rank:03d}.jsonl")

    @torch.no_grad()
    def _maybe_write_transformed_y_hist(self, step, beta):
        if self._transformed_y_hist_log_every <= 0:
            return
        if step is None or step <= 0 or step % self._transformed_y_hist_log_every != 0:
            return

        out_path = self._transformed_y_hist_output_path()
        if out_path is None:
            return

        edges = self._transformed_y_hist_edges
        counts = torch.zeros(self._transformed_y_hist_bins, dtype=torch.int64)
        total = 0
        in_range = 0
        finite_sum = 0.0
        finite_min = None
        finite_max = None
        factor = self._transformed_y_hist_factor(beta)

        for state in self.state.values():
            if not isinstance(state, dict) or "y_t" not in state:
                continue
            y = state["y_t"]
            if hasattr(y, "to_local"):
                y = y.to_local()
            if not isinstance(y, torch.Tensor):
                continue
            vals = y.detach().reshape(-1).to(dtype=torch.float32)
            vals = vals[torch.isfinite(vals)]
            if vals.numel() == 0:
                continue
            if factor != 1.0:
                vals = vals * factor
            total += int(vals.numel())
            finite_sum += float(vals.sum().item())
            cur_min = float(vals.min().item())
            cur_max = float(vals.max().item())
            finite_min = cur_min if finite_min is None else min(finite_min, cur_min)
            finite_max = cur_max if finite_max is None else max(finite_max, cur_max)
            vals_cpu = vals.cpu().to(dtype=torch.float64)
            hist = torch.histc(
                vals_cpu,
                bins=self._transformed_y_hist_bins,
                min=self._transformed_y_hist_min,
                max=self._transformed_y_hist_max,
            ).to(dtype=torch.int64)
            counts += hist
            in_range += int(((vals_cpu >= self._transformed_y_hist_min) & (vals_cpu <= self._transformed_y_hist_max)).sum().item())

        if total == 0:
            return

        record = {
            "schema": "plainlm_adam_nsr_transformed_y_hist_v1",
            "time": time.time(),
            "rank": int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "run_name": self._transformed_y_hist_run_name,
            "step": int(step),
            "total_training_steps": self._transformed_y_hist_total_steps,
            "beta": float(beta),
            "value": "adam_nsr_state_y_t",
            "scale": self._transformed_y_hist_scale,
            "formula_if_adam_m_v": "((v-m^2)/m^2)*((1-beta)/beta)",
            "formula_requested_beta_over_one_minus_beta": self._transformed_y_hist_scale == "beta_over_one_minus_beta",
            "hist_min": self._transformed_y_hist_min,
            "hist_max": self._transformed_y_hist_max,
            "hist_bins": self._transformed_y_hist_bins,
            "edges": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in counts.tolist()],
            "total_finite": int(total),
            "in_range": int(in_range),
            "mean": finite_sum / float(total),
            "min": finite_min,
            "max": finite_max,
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_step_table_counts = None
        total_step_table = None
        global_step = None

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["betas"][0]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            y_max = group["y_max"]
            m_dtype = group.get("moment1_dtype", None)
            m_quant = group.get("moment1_quant", "per_column")
            y_dtype = group.get("moment2_dtype", None)
            compute_dtype = group.get("compute_dtype", None) or torch.float32
            explicit_y_table = group.get("y_table", None)
            skip_update_on_bad_y = group.get("skip_update_on_bad_y", False)
            use_y_table = _uses_y_table(y_dtype, explicit_y_table)
            if use_y_table:
                y_table = self._table_tensor(torch.empty(0, device=group["params"][0].device), y_dtype, explicit_y_table)
                y_floor = float(y_table[0].item())
                if explicit_y_table is not None:
                    y_clip_max = float(y_table[-1].item())
                else:
                    y_clip_max = min(float(y_max), float(y_table[-1].item()))
            else:
                y_table = None
                y_floor = 0.0
                y_clip_max = float(y_max)

            step_table_counts = None

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamNSR does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    state["y_t"] = torch.full_like(
                        p,
                        fill_value=y_floor,
                        dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    )

                m = state["exp_avg"]
                y = state["y_t"]
                state["step"] += 1
                step = state["step"]
                global_step = step

                grad_work = grad.detach().to(dtype=compute_dtype)
                param_work = p.data.detach().to(dtype=compute_dtype)

                # shadow states are already fake-quantized on writeback
                m_old_work = m.to(dtype=compute_dtype)
                y_prev_work = y.to(dtype=compute_dtype)

                m_work = m_old_work.clone()
                m_work.mul_(beta).add_(grad_work, alpha=1.0 - beta)

                if step == 1:
                    y_work = torch.ones_like(y_prev_work, dtype=compute_dtype)
                    bad_mask = torch.zeros_like(y_work, dtype=torch.bool)
                else:
                    x_t = m_old_work / m_work
                    y_raw = beta * x_t.square() * y_prev_work + (1.0 - x_t).square()
                    bad_mask = (~torch.isfinite(y_raw)) | (y_raw < 0.0) | (y_raw > y_clip_max)
                    y_work = torch.nan_to_num(y_raw, nan=y_clip_max, posinf=y_clip_max, neginf=0.0)
                    y_work = y_work.clamp(min=0.0, max=y_clip_max)

                if use_y_table:
                    y_value_fp32, y_counts = self._quantize_y_table(
                        y_work,
                        y_clip_max,
                        y_dtype,
                        explicit_y_table=explicit_y_table,
                        return_counts=True,
                    )
                    if step_table_counts is None:
                        step_table_counts = y_counts
                    else:
                        step_table_counts.add_(y_counts)
                    total_step_table = y_table
                else:
                    y_value_fp32 = y_work
                y_store = _fake_quantize_state(y_value_fp32, y_dtype)
                y_work_quantized = y_store.to(dtype=compute_dtype)

                m_store = _fake_quantize_state(m_work, m_dtype, group_mode=m_quant)
                m.copy_(m_store.to(dtype=m.dtype))
                y.copy_(y_store.to(dtype=y.dtype))

                bc = 1.0 - beta**step
                m_hat = m_work / bc
                r_work = (beta / (1.0 - beta)) * y_work_quantized
                scale = torch.sqrt((1.0 + r_work) / bc)
                # Adam-equivalent denominator: sqrt(v_hat) = |m_t| * sqrt((1 + r_t) / bc).
                if eps == 0.0:
                    # Algebraically cancel |m_t| before evaluating the
                    # no-epsilon update.  This is identical for m_t != 0 and
                    # extends continuously to zero through sign(0) == 0,
                    # without introducing an additive safety constant.
                    update = torch.sign(m_work) / (bc * scale)
                else:
                    denom = m_work.abs() * scale + eps
                    update = m_hat / denom
                if step > 1 and skip_update_on_bad_y:
                    update = torch.where(bad_mask, torch.zeros_like(update), update)

                if weight_decay != 0:
                    param_work.mul_(1.0 - lr * weight_decay)
                param_work.add_(update, alpha=-lr)
                p.data.copy_(param_work.to(dtype=p.data.dtype))

            if use_y_table and step_table_counts is not None:
                if total_step_table_counts is None:
                    total_step_table_counts = step_table_counts
                else:
                    total_step_table_counts.add_(step_table_counts)

        if total_step_table_counts is not None and total_step_table is not None and global_step is not None:
            self._update_table_hit_window(global_step, total_step_table, total_step_table_counts)
        if global_step is not None:
            beta = self.param_groups[0]["betas"][0]
            self._maybe_write_transformed_y_hist(global_step, beta)

        return loss
