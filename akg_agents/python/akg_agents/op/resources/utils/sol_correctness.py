import random
from typing import Optional, Tuple
import torch

class ToleranceSpec:
    def __init__(self, max_atol=1e-2, max_rtol=1e-2, required_matched_ratio=0.99, max_error_cap=None, allow_negative_inf=False):
        self.max_atol = max_atol
        self.max_rtol = max_rtol
        self.required_matched_ratio = required_matched_ratio
        self.max_error_cap = max_error_cap
        self.allow_negative_inf = allow_negative_inf

class Correctness:
    def __init__(self, max_absolute_error=0.0, max_relative_error=0.0, has_nan=False, has_inf=False):
        self.max_absolute_error = max_absolute_error
        self.max_relative_error = max_relative_error
        self.has_nan = has_nan
        self.has_inf = has_inf

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, PyTorch CPU and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output."""
    ref_nonfinite = ~torch.isfinite(ref_tensor)
    sol_nonfinite = ~torch.isfinite(sol_tensor)

    if allow_negative_inf:
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_nonfinite = ref_nonfinite & ~both_neg_inf
        sol_nonfinite = sol_nonfinite & ~both_neg_inf

    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()
    if has_nonfinite:
        has_nan = (
            torch.isnan(sol_tensor).any().item() or torch.isnan(ref_tensor).any().item()
        )
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None

def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec
) -> Tuple[Correctness, bool]:
    """Compute numerical error between *output* and *reference*."""
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    allow_neg_inf = tolerance.allow_negative_inf

    infs_nans = check_tensor_sanity(x, y, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return infs_nans, True

    if allow_neg_inf:
        both_neg_inf = (x == float("-inf")) & (y == float("-inf"))
        finite_mask = ~both_neg_inf
        x = x[finite_mask]
        y = y[finite_mask]

    abs_error = torch.abs(x - y)
    total_elements = abs_error.numel()
    if total_elements == 0:
        return Correctness(), False

    max_abs = float(abs_error.max().item())

    tol_bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)
    exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)
    del tol_bound

    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds_tol = True

    rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)
    max_rel = float(rel_error.max().item())

    return Correctness(
        max_absolute_error=max_abs, max_relative_error=max_rel
    ), exceeds_tol
