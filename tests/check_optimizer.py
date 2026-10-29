"""CPU checks for packaged optimizer defaults and update semantics; no training."""
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_codebook():
    # Independent expected values, deliberately not read from the implementation.
    return (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def check_update_order(module, momentum_dtype, eps):
    beta = 0.9
    correction = 1 - beta**2
    parameter = torch.nn.Parameter(torch.zeros(3))
    gradient = torch.tensor([0.3, 0.43, -0.03])
    old_momentum = torch.tensor([0.12, -0.2, 0.006])
    old_ratio = torch.tensor([1.0, 2.0, 4.0])
    optimizer = module.AdamNSR(
        [parameter], lr=1, betas=(beta, beta), eps=eps,
        moment1_dtype=momentum_dtype, moment2_dtype="fp32", compute_dtype="fp32",
    )
    assert optimizer.param_groups[0]["y_table"] == reference_codebook()
    optimizer.state[parameter] = {
        "step": 1, "exp_avg": old_momentum.clone(), "y_t": old_ratio.clone(),
    }
    parameter.grad = gradient
    fresh = old_momentum.clone().mul_(beta).add_(gradient, alpha=1-beta)
    x = old_momentum / fresh
    candidate = (beta*x.square()*old_ratio + (1-x).square()).clamp(0, 64)
    table = torch.tensor(reference_codebook())
    quantized = table[(candidate[:, None]-table[None, :]).abs().argmin(1)]
    stored_momentum = (
        module._fake_quantize_state(fresh, torch.float8_e4m3fn)
        if momentum_dtype == "fp8_e4m3" else fresh
    )
    expected = (fresh/correction) / (
        fresh.abs()*torch.sqrt((1 + beta/(1-beta)*quantized)/correction) + eps
    )
    optimizer.step()
    torch.testing.assert_close(-parameter.detach(), expected, rtol=2e-6, atol=2e-7)
    assert torch.equal(optimizer.state[parameter]["y_t"], quantized)
    assert torch.equal(optimizer.state[parameter]["exp_avg"], stored_momentum)
    print(momentum_dtype, "default codebook and update order PASS")


def check_zero_epsilon(module):
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = module.AdamNSR(
        [parameter], lr=1, betas=(0.9, 0.9), eps=0,
        moment1_dtype="fp32", moment2_dtype="fp32",
    )
    assert optimizer.param_groups[0]["y_table"] == reference_codebook()
    optimizer.state[parameter] = {
        "step": 1, "exp_avg": torch.tensor([0.0, 0.12]), "y_t": torch.ones(2),
    }
    parameter.grad = torch.tensor([0.0, 0.3])
    optimizer.step()
    assert torch.isfinite(parameter).all() and parameter[0] == 0
    print("zero-epsilon default and zero momentum PASS")


def check_overrides(module):
    parameter = torch.nn.Parameter(torch.ones(2))
    for value, expected in [(None, None), ([0.5, 1, 2], (0.5, 1, 2))]:
        optimizer = module.AdamNSR([parameter], y_table=value)
        assert optimizer.param_groups[0]["y_table"] == expected
    print("explicit custom table and legacy None PASS")


if __name__ == "__main__":
    standard = load_module("standard", ROOT/"optimizer/adam_nsr.py")
    zero = load_module("zero", ROOT/"pretrain/optim/adam_nsr_eps0.py")
    check_update_order(standard, "fp32", 1e-6)
    # A larger synthetic epsilon makes incorrect use of stored FP8 momentum visible.
    check_update_order(standard, "fp8_e4m3", 0.03)
    check_zero_epsilon(zero)
    check_overrides(standard)
    reference = (ROOT/"optimizer/adam_nsr.py").read_bytes()
    for rel in ["pretrain/optim/adam_nsr.py", "posttrain/verl/utils/adam_nsr.py", "cifar/adam_nsr.py"]:
        assert (ROOT/rel).read_bytes() == reference, rel
    print("optimizer copies identical PASS")
