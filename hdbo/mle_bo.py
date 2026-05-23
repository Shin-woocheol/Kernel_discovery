"""
Self-contained MLE kernel framework for HDBO.
Copied and adapted from blackbox_opt.mle_bo to avoid cross-package imports.
"""
import math
import os
import random
import re
import time
import traceback as _traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import gpytorch
import torch

from .llm_client import LLMClient
from .bo import (
    DSP_KERNEL_CODE,
    MATERN32_KERNEL_CODE,
    MATERN52_KERNEL_CODE,
    RQ_KERNEL_CODE,
    SPHERICAL_KERNEL_CODE,
    CYLINDRICAL_KERNEL_CODE,
    LINEAR_KERNEL_CODE,
    PERIODIC_KERNEL_CODE,
    DISCOVERY_1_KERNEL_CODE,
    DISCOVERY_2_KERNEL_CODE,
    RQ_SINGLE_KERNEL_CODE,
    DSP_SINGLE_KERNEL_CODE,
    MATERN52_SINGLE_KERNEL_CODE,
    _initialize_model,
    _fit_model,
    _kernel_from_code,
)
from .loocv import loocv_nlpd_loss, loocv_crps_loss, rank_weighted_loss


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_instance):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_instance)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def compute_log_prior(model, likelihood):
    """
    Aggregates the log-prior probabilities of all hyperparameters.
    This acts as our complexity penalty (Occam's razor).
    named_priors() yields (name, module, prior, closure, setting_closure).
    Use closure(module) to get the value for prior.log_prob.
    """
    log_prior = 0.0
    for _, module, prior, closure, _ in model.named_priors():
        log_prior += prior.log_prob(closure(module)).sum()
    for _, module, prior, closure, _ in likelihood.named_priors():
        log_prior += prior.log_prob(closure(module)).sum()
    return log_prior


def _model_selection_score(loss, model, n, criterion="loss"):
    """Convert NLL loss to selection score. Lower is better.

    Note: `loss` is per-sample NLL (gpytorch divides by N internally).
    - "bic": legacy, uses per-sample loss directly (penalty-dominated).
    - "bic_correct": proper BIC = 2*N*loss + k*log(n), scales loss to total NLL.
    """
    if criterion == "bic":
        k = sum(p.numel() for p in model.parameters())
        return 2.0 * loss + k * math.log(n)
    if criterion == "bic_correct":
        k = sum(p.numel() for p in model.parameters())
        return 2.0 * n * loss + k * math.log(n)
    return loss


def _crps_bic_score(crps: float, n: int, model) -> float:
    """BIC-style criterion with mean LOOCV CRPS in place of per-sample NLL (``bic_correct`` scaling).

    score = 2*n*CRPS_mean + k*log(n), with k = number of parameters.
    """
    k = sum(p.numel() for p in model.parameters())
    return 2.0 * n * crps + k * math.log(n)


def _verify_kernel_agnostic_to_n(kernel_instance, x_sample, device):
    """Verify kernel works for arbitrary N, M (number of test points). Rejects hardcoded sizes.

    Returns (ok: bool, diag: str). diag is empty string on success.
    """
    if x_sample.dim() == 1:
        x_sample = x_sample.unsqueeze(-1)
    n_avail = x_sample.shape[0]
    if n_avail < 2:
        return True, ""
    x_sample = x_sample.to(device)
    D = x_sample.shape[1]
    # 2D cases: basic shape checks
    cases_2d = [(1, 1), (1, min(5, n_avail)), (min(5, n_avail), 1), (min(3, n_avail), min(7, n_avail))]
    # 3D batched cases: simulate BoTorch acqf evaluation (num_restarts × q × d vs n_train × d)
    # BoTorch passes x1=(batch, q, D) and x2=(batch, n_train, D) to kernel.forward
    cases_3d = [
        (min(4, n_avail), min(3, n_avail)),   # simulates (num_restarts=4, q, D) vs (n_train, D)
        (min(2, n_avail), min(7, n_avail)),   # different batch vs n
    ]

    def _check_one(x1, x2, label):
        try:
            out = kernel_instance(x1, x2).to_dense()
            n1, n2 = x1.shape[-2], x2.shape[-2]
            if out.shape[-2:] != (n1, n2):
                return (
                    f"kernel output shape is N-dependent: "
                    f"called with x1.shape={tuple(x1.shape)}, x2.shape={tuple(x2.shape)}, "
                    f"expected output shape ({n1},{n2}), got {tuple(out.shape[-2:])}. "
                    f"Common cause: torch.eye(x1.size(0)) or torch.eye(x2.size(0)) hardcodes the "
                    f"diagonal size. The kernel must support any (N1, M) output shape — "
                    f"use vectorized pairwise ops via torch.cdist (for distances) or matmul "
                    f"(for inner products), and never construct matrices whose size is tied to N or M."
                )
        except Exception as e:
            e_str = str(e)
            e_lower = e_str.lower()

            # Extract only user kernel frames (exec'd code shows as "<string>"),
            # stripping all GPyTorch/library internals.
            tb_lines = _traceback.format_exc().splitlines()
            user_frames = []
            for i, line in enumerate(tb_lines):
                if "<string>" in line:
                    user_frames.append(line.strip())
                    if i + 1 < len(tb_lines) and not tb_lines[i + 1].strip().startswith("File "):
                        user_frames.append("  " + tb_lines[i + 1].strip())
            user_tb = ("\nUser kernel traceback:\n  " + "\n  ".join(user_frames)) if user_frames else ""

            # --- Targeted hints per error pattern ---
            if "expected shape" in e_lower and "gpytorch" in e_lower:
                # GPyTorch internal shape validation: forward() returned wrong shape.
                # Replace the confusing "likely a bug in GPyTorch" message with a clear explanation.
                import re as _re
                m = _re.search(r"expected shape of the kernel was ([\w\[\], ]+), but got ([\w\[\], ]+)\.", e_str)
                if m:
                    exc_str = f"forward() returned shape {m.group(2).strip()} but expected {m.group(1).strip()}"
                else:
                    exc_str = f"{type(e).__name__}: forward() returned wrong shape"
                hint = (
                    "forward() must return exactly (N1, N2) for 2D inputs or (B, N1, N2) for batched 3D inputs. "
                    "Do not add extra dimensions. Common causes: "
                    "torch.eye(N) forces square output; "
                    "extra unsqueeze in forward() inflates dims; "
                    "calling a sub-kernel that expands shape."
                )
            elif "broadcast" in e_lower or ("shape" in e_lower and "match" in e_lower):
                exc_str = f"{type(e).__name__}: {e}"
                hint = (
                    "Cause: intermediate tensor shapes are incompatible for broadcast. "
                    "For pairwise distances, use `torch.cdist(x1_scaled, x2_scaled, p=2)` after "
                    "rescaling by `self.lengthscale` — it returns shape (..., N1, N2) directly without "
                    "building a (..., N1, N2, D) intermediate. For inner products, use "
                    "`x1 @ x2.transpose(-1, -2)`. Never use torch.eye(N) or any size tied to N1/N2."
                )
            elif "size" in e_lower and ("mismatch" in e_lower or "must match" in e_lower):
                exc_str = f"{type(e).__name__}: {e}"
                hint = "Cause: matrix multiply or element-wise op with incompatible sizes. Check all intermediate tensor shapes."
            else:
                exc_str = f"{type(e).__name__}: {e}"
                hint = "Ensure forward() handles arbitrary (N1, N2) input sizes without hardcoding N."

            return (
                f"x1.shape={tuple(x1.shape)}, x2.shape={tuple(x2.shape)}: "
                f"{exc_str}. {hint}{user_tb}"
            )
        return None

    for n1, n2 in cases_2d:
        if n1 > n_avail or n2 > n_avail:
            continue
        diag = _check_one(x_sample[:n1], x_sample[:n2], "2D")
        if diag:
            return False, diag

    # 3D batched: x1 shape (batch, n1, D), x2 shape (batch, n2, D)
    for n1, n2 in cases_3d:
        if n1 > n_avail or n2 > n_avail:
            continue
        x1_3d = x_sample[:n1].unsqueeze(0)   # (1, n1, D)
        x2_3d = x_sample[:n2].unsqueeze(0)   # (1, n2, D)
        diag = _check_one(x1_3d, x2_3d, "3D batched")
        if diag:
            return False, diag

    return True, ""


def _verify_kernel_differentiable(kernel_instance, x_sample, device):
    """Verify kernel output is differentiable w.r.t. inputs.

    Required for gradient-based acquisition optimization (gen_candidates_torch).
    Returns (ok: bool, diag: str). diag is empty string on success.
    """
    try:
        x = x_sample[:3].detach().clone().requires_grad_(True)
        out = kernel_instance(x, x).to_dense()
        out.sum().backward()
        if x.grad is None:
            return False, (
                "Kernel output is not differentiable w.r.t. inputs (x.grad is None after backward()). "
                "Common causes: using .detach(), .numpy(), np.array(), or torch.no_grad() inside forward(). "
                "All operations in forward() must be pure differentiable PyTorch ops."
            )
        return True, ""
    except Exception as e:
        return False, (
            f"Kernel raised {type(e).__name__}: {e} during differentiability check (backward pass). "
            f"Common causes: using .detach(), .numpy(), np.array(), or non-differentiable ops inside forward(). "
            f"All operations must be pure differentiable PyTorch ops — no NumPy, no .item() inside computations."
        )


def _verify_kernel_psd(kernel_instance, x_sample, device):
    """Verify kernel is Positive Semi-Definite by conducting Cholesky decomposition."""
    if x_sample.dim() == 1:
        x_sample = x_sample.unsqueeze(-1)
    x_sample = x_sample.to(device)
    try:
        # Evaluate kernel matrix on sample points
        # Using a small jitter is common practice for numerical stability, 
        # but here we want to ensure it's theoretically PSD.
        # However, to be practical, we check if Cholesky succeeds on K + 1e-6*I
        K = kernel_instance(x_sample, x_sample).to_dense()
        torch.linalg.cholesky(K + 1e-6 * torch.eye(K.shape[-1], device=device, dtype=K.dtype))
        return True
    except Exception:
        return False


def evaluate_kernel_code(
    kernel_code_str,
    train_x,
    train_y,
    training_iterations=50,
    model_selection="loss",
    state_dict=None,
    skip_fit=False,
    timeout=120.0,
    bounds=None,
    strict_load=True,
    eval_subsample=1.0,
    complexity_lambda=0.0,
):
    start_eval = time.perf_counter()
    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
    device = train_x.device
    n = train_x.shape[0]
    input_dim = train_x.shape[1] if train_x.dim() > 1 else 1
    if bounds is None:
        bounds = [(0.0, 1.0)] * input_dim

    # --- Data subsampling for fitting (score on full data) ---
    full_train_x, full_train_y = train_x, train_y
    if 0.0 < eval_subsample < 1.0 and n > 2 * input_dim:
        sub_n = max(int(n * eval_subsample), 2 * input_dim)
        if sub_n < n:
            perm = torch.randperm(n, device=device)[:sub_n]
            train_x = train_x[perm]
            train_y = train_y[perm]
            n = sub_n

    # Run all model verification and initialization inside the ThreadPool to catch hangs early
    def _execute_full_pass():
        # --- Stage 1: parse & build kernel ---
        try:
            kernel_instance = _kernel_from_code(
                kernel_code_str, input_dim, device, train_x.dtype, bounds=bounds, wrap_scale=False
            )
        except SyntaxError as e:
            return float("inf"), None, "syntax_error", _traceback.format_exc(), 0.0, 0.0
        except KeyError as e:
            msg = str(e)
            # KeyError from _kernel_from_code can be: (1) class not found, or (2) gpytorch
            # registration conflict (e.g. "attribute 'lengthscale' already exists")
            if "EvolvedKernel" in msg or "Kernel subclass" in msg:
                return float("inf"), None, "no_kernel_class", _traceback.format_exc(), 0.0, 0.0
            else:
                return float("inf"), None, "instantiation_error", _traceback.format_exc(), 0.0, 0.0
        except Exception as e:
            return float("inf"), None, "instantiation_error", _traceback.format_exc(), 0.0, 0.0

        if kernel_instance is None:
            return float("inf"), None, "instantiation_error", "kernel_from_code returned None", 0.0, 0.0

        # --- Stage 2: sanity checks ---
        _t_val = time.perf_counter()
        agnostic_ok, agnostic_diag = _verify_kernel_agnostic_to_n(kernel_instance, train_x[:10], device)
        if not agnostic_ok:
            return float("inf"), None, "agnostic_check", agnostic_diag, time.perf_counter() - _t_val, 0.0

        grad_ok, grad_diag = _verify_kernel_differentiable(kernel_instance, train_x[:3], device)
        if not grad_ok:
            return float("inf"), None, "grad_check", grad_diag, time.perf_counter() - _t_val, 0.0

        if not _verify_kernel_psd(kernel_instance, train_x[:10], device):
            return float("inf"), None, "psd_check", "Cholesky decomposition failed (not PSD)", time.perf_counter() - _t_val, 0.0
        t_validation = time.perf_counter() - _t_val

        # --- Stage 3: model init & fit ---
        _t_fit = time.perf_counter()
        try:
            Y_train = train_y
            if Y_train.dim() == 1:
                Y_train = Y_train.unsqueeze(-1)
            model = _initialize_model(
                train_x,
                Y_train,
                base_kernel=kernel_code_str,
                device=device,
                dtype=train_x.dtype,
                bounds=bounds,
            )
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=strict_load)
            final_loss = _fit_model(model, return_loss=True, skip_fit=skip_fit)
            return final_loss, model, None, None, t_validation, time.perf_counter() - _t_fit
        except Exception as e:
            return float("inf"), None, "fit_error", _traceback.format_exc(), t_validation, time.perf_counter() - _t_fit

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    future = executor.submit(_execute_full_pass)
    try:
        final_loss, model, fail_reason, tb_str, t_validation_sec, t_fit_sec = future.result(timeout=timeout)
        executor.shutdown(wait=True)
    except (concurrent.futures.TimeoutError, TimeoutError):
        print(f"    [Evaluation] FAILED: exceeded timeout of {timeout}s")
        executor.shutdown(wait=False, cancel_futures=True)
        return float("inf"), None, 0.0, "timeout", f"exceeded timeout of {timeout}s", float("inf"), {}
    except Exception as e:
        print(f"    [Evaluation] FAILED: {e}")
        executor.shutdown(wait=False, cancel_futures=True)
        return float("inf"), None, 0.0, "outer_error", _traceback.format_exc(), float("inf"), {}

    if final_loss is None or math.isnan(final_loss) or model is None:
        return float("inf"), None, 0.0, fail_reason or "nan_loss", tb_str, float("inf"), {}

    # --- Stage 4: model selection scoring ---
    _t_score = time.perf_counter()
    try:
        if model_selection == "nlpd":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = loocv_nlpd_loss(model, model.likelihood, train_x, train_y).item()
        elif model_selection == "crps":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = loocv_crps_loss(model, model.likelihood, train_x, train_y).item()
        elif model_selection == "rank_based":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = rank_weighted_loss(model, model.likelihood, train_x, train_y).item()
        elif model_selection == "loss_prior":
            score = _model_selection_score(final_loss, model, n, "loss")
            score = score - compute_log_prior(model, model.likelihood).item()
        elif model_selection == "nlpd_prior":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = loocv_nlpd_loss(model, model.likelihood, train_x, train_y).item()
            score = score - compute_log_prior(model, model.likelihood).item()
        elif model_selection == "crps_prior":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = loocv_crps_loss(model, model.likelihood, train_x, train_y).item()
            score = score - compute_log_prior(model, model.likelihood).item()
        elif model_selection == "rank_based_prior":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = rank_weighted_loss(model, model.likelihood, train_x, train_y).item()
            score = score - compute_log_prior(model, model.likelihood).item()
        elif model_selection == "crps_complexity":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            with torch.no_grad():
                score = loocv_crps_loss(model, model.likelihood, train_x, train_y).item()
            if complexity_lambda > 0:
                n_params = sum(p.numel() for p in model.parameters())
                score += complexity_lambda * n_params / train_x.shape[0]
        elif model_selection == "crps_bic":
            train_x = model.train_inputs[0]
            train_y = model.train_targets.squeeze(-1)
            n_obs = train_x.shape[0]
            with torch.no_grad():
                crps = loocv_crps_loss(model, model.likelihood, train_x, train_y).item()
            score = _crps_bic_score(crps, n_obs, model)
        else:
            score = _model_selection_score(final_loss, model, n, model_selection)
    except Exception:
        return float("inf"), None, 0.0, "fit_error", _traceback.format_exc(), float("inf"), {}
    t_scoring_sec = time.perf_counter() - _t_score

    # Measure forward pass time for time complexity consideration
    try:
        start_time = time.perf_counter()
        with torch.no_grad():
            # A dummy forward pass on a reasonably sized matrix to check speed
            test_x = torch.randn(min(100, n), input_dim, device=device, dtype=train_x.dtype)
            model.eval()
            model(test_x)
        eval_time = time.perf_counter() - start_time
    except Exception:
        eval_time = 0.0

    total_eval_time = time.perf_counter() - start_eval
    print(f"    [Evaluation] score={score:.4f}, nll={final_loss:.4f}, eval_time={total_eval_time:.4f}s")

    timing_breakdown = {
        "validation_sec": round(t_validation_sec, 4),
        "fit_sec": round(t_fit_sec, 4),
        "scoring_sec": round(t_scoring_sec, 4),
    }
    return score, model, eval_time, None, None, final_loss, timing_breakdown


def mutate_code_with_llm(current_code, current_loss, current_time=None, client: LLMClient | None = None, max_compositions=3, temperature=1.0):
    if client is None:
        return current_code

    time_info = f"\nCurrent Evaluation Time: {current_time:.6f}s (lower is faster/better)" if current_time is not None else ""
    prompt = f"""You are an evolutionary coding algorithm. Optimize a GP covariance kernel by writing a class `EvolvedKernel`.
Current NLL: {current_loss:.4f} (lower is better).{time_info}

Current Code:
```python
{current_code}
```

CRITICAL DISCOVERY INSTRUCTIONS:
1. HIGH-DIMENSIONAL ROBUSTNESS (CRITICAL): In higher dimensions, standard kernels fail due to boundary-seeking behavior and distance collapse.
2. STRUCTURAL ASSUMPTIONS FOR HDBO: You are highly encouraged to build kernels that impose structural assumptions on the high-dimensional space.
3. You must write a class named `EvolvedKernel` that inherits from `gpytorch.kernels.Kernel`.
4. Keep code CONCISE: avoid verbose comments, redundant docstrings, or unnecessary boilerplate.
5. Keep computation EFFICIENT: use optimized vectorized operations.

CRITICAL RULES (kernels frequently fail due to these):
- FORWARD SIGNATURE: Always `def forward(self, x1, x2, diag=False, **params)` — GPyTorch passes extra kwargs like `last_dim_is_batch`.
- SHAPE INVARIANT: `forward()` MUST return exactly `(..., N1, N2)` for ANY combination of N1 and N2 (e.g., N1=3, N2=7 non-square). `x1` has shape `(..., N1, D)` and `x2` has shape `(..., N2, D)`. Only use `self.attribute` values defined in `__init__` inside `forward()`.
- PAIRWISE DISTANCE: Use `torch.cdist(x1_scaled, x2_scaled, p=2)` after rescaling by `self.lengthscale`. Returns shape (..., N1, N2). For inner-product kernels use `x1 @ x2.transpose(-1, -2)`. Do NOT subtract broadcast-unsqueezed tensors to form a (..., N1, N2, D) intermediate — CUDA OOM.
- ARD LENGTHSCALE: `self.lengthscale` has shape (*batch, 1, D). Rescale via `x1s = x1 / self.lengthscale`, `x2s = x2 / self.lengthscale`, then `dist2 = torch.cdist(x1s, x2s, p=2).pow(2)` → shape (..., N1, N2).
- LENGTHSCALE REGISTRATION: Choose ONE approach. APPROACH A (recommended): set `has_lengthscale = True` and pass `lengthscale_prior`/`lengthscale_constraint` to `super().__init__()` — do NOT also call `register_parameter("raw_lengthscale", ...)`. APPROACH B: do NOT set `has_lengthscale = True`, manually register `raw_lengthscale` and define a `@property`. Never mix both.
- PARAMETER REGISTRATION ORDER: Always call `register_parameter(name, ...)` BEFORE `register_constraint(name, ...)` for the same parameter.
- DEVICE: Tensors created in `__init__` via `torch.tensor(...)` stay on CPU. Use `register_buffer(...)` so they move with the model, or call `.to(x1.device)` in `forward()`. Any new tensor created inside `forward()` must be on `x1.device` (e.g., `torch.zeros(..., device=x1.device)`).
- NO IN-PLACE OPERATIONS: Never use `+=`, `*=`, `.add_()`, `.mul_()`, `.clamp_()`, `.pow_()`, `.exp_()`, `.sqrt_()`. Use `x = x + y`, `x = x.clamp(...)` etc. In-place ops break the autograd graph.
- NO DETACH/NUMPY IN forward(): Never call `.detach()`, `.numpy()`, `np.array()`, `.item()`, or use `torch.no_grad()` inside `forward()`. These break the autograd graph and cause gradient failures during acquisition optimization.
- DIAG: When `diag=True`, return `covar.diagonal(dim1=-2, dim2=-1)`.

Be creative! The evaluation environment will automatically reject mathematically invalid (non-PSD) proposals, so you are free to experiment with highly unconventional mathematics.

Return ONLY valid Python code in a ```python block. Do not include any explanations."""

    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=8192)
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1) if match else text
    except Exception as e:
        print(f"    [Generation] Error: {e}")
        return current_code


def fix_code_with_llm(failed_code: str, fail_reason: str, tb_str: str, client: LLMClient | None = None, temperature: float = 0.7) -> str:
    """Send failed kernel code + error back to LLM for repair."""
    if client is None:
        return failed_code

    # Keep last 2000 chars of traceback (most informative part)
    tb_truncated = tb_str[-2000:] if len(tb_str) > 2000 else tb_str


    prompt = f"""You are a GP kernel debugging assistant. The following `EvolvedKernel` class failed evaluation with this error:

{tb_truncated}

Failed Code:
```python
{failed_code}
```
Read the error carefully and fix the code. Return ONLY the fixed Python code in a ```python block. Keep the class named `EvolvedKernel`. Do not include explanations."""

#     prompt = f"""You are a GP kernel debugging assistant. The following `EvolvedKernel` class failed evaluation with this error:

# {tb_truncated}

# Failed Code:
# ```python
# {failed_code}
# ```


# Read the error carefully and fix ONLY the specific issue. Do not rewrite unrelated parts.

# CRITICAL RULES — also check that your fix does not violate these:
# - SHAPE INVARIANT: `forward()` MUST return exactly `(..., N1, N2)` for ANY N1, N2 (non-square too). `x1` is `(..., N1, D)`, `x2` is `(..., N2, D)`.
#   FORBIDDEN patterns that cause shape errors:
#     - `torch.eye(x1.shape[0])` or `torch.eye(N)` — creates (N,N) not (N1,N2). ❌
#     - Any matrix construction that uses `x1.shape[-2]` as both dimensions. ❌
#     - `x1.unsqueeze(-2) - x2.unsqueeze(-3)` — materializes (...,N1,N2,D), CUDA OOM. ❌
#   SAFE patterns: `torch.cdist(x1s, x2s)` → (...,N1,N2). `x1 @ x2.transpose(-1,-2)` → (...,N1,N2).
# - DEVICE: tensors created in `__init__` must use `register_buffer(...)` or `.to(x1.device)` in `forward()`.
# - NO IN-PLACE: Never `+=`, `*=`, `.add_()`, etc. Use `x = x + y`.
# - NO DETACH/NUMPY in `forward()`: No `.detach()`, `.numpy()`, `.item()`, `torch.no_grad()`.
# - PARAMETER REGISTRATION ORDER: `register_parameter(name, ...)` BEFORE `register_constraint(name, ...)`.
# - FORWARD SIGNATURE: `def forward(self, x1, x2, diag=False, **params)`.

# Return ONLY the fixed Python code in a ```python block. Keep the class named `EvolvedKernel`. Do not include explanations."""

    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=8192)
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1) if match else failed_code
    except Exception as e:
        print(f"    [Fix] LLM error: {e}")
        return failed_code


def _multi_kernel_composition_impl(population_subset, client, max_compositions=3):
    prompt = f"Top {len(population_subset)} EvolvedKernel classes:\n\n"
    for i, ind in enumerate(population_subset):
        prompt += f"--- Kernel {i} (Loss: {ind['loss']:.4f}) ---\n{ind['code']}\n\n"
    prompt += f"""Compose a subset into a SINGLE new `EvolvedKernel` class.
- YOU decide which kernels to combine: choose k (2 <= k <= {len(population_subset)}) from the above. Pick the ones you think will work best together (consider loss, structure, complementarity).
- Addition or Multiplication. LIMIT the total number of compositions (operators like + or *) to AT MOST {max_compositions}. For example, if max_compositions=1, only K1 + K2 is allowed.
- __init__ MUST accept ard_num_dims: def __init__(self, ard_num_dims: int, **kwargs):
- USE PRIORS AND CONSTRAINTS for hyperparameters (LogNormalPrior, GreaterThan, Positive). When using `has_lengthscale=True`, pass `lengthscale_prior`/`lengthscale_constraint` to `super().__init__()`. Register all other parameters in __init__.
- Use ASCII only: math.pi not π, * not ×.
- Keep code CONCISE.

CRITICAL RULES (compositions frequently fail due to these):
- CLASS NAME: The class MUST be named exactly `EvolvedKernel`.
- LENGTHSCALE: Choose ONE approach. APPROACH A (recommended): set `has_lengthscale = True` and pass `lengthscale_prior`/`lengthscale_constraint` to `super().__init__()` — do NOT also call `register_parameter("raw_lengthscale", ...)`. APPROACH B: do NOT set `has_lengthscale = True`, manually register `raw_lengthscale` and define a `@property`. NEVER write `self.lengthscale = <anything>` — raises `KeyError: "attribute 'lengthscale' already exists"`.
- PARAMETER REGISTRATION ORDER: Always call `register_parameter(name, ...)` BEFORE `register_constraint(name, ...)` for the same parameter.
- SHAPE INVARIANT: `forward()` MUST return exactly `(..., N1, N2)` for ANY N1, N2 (e.g., N1=3, N2=7 non-square). `x1` is `(..., N1, D)`, `x2` is `(..., N2, D)`. Only use `self.attribute` values defined in `__init__` inside `forward()`. Mentally verify output shape with N1=3, N2=7.
- NO SELF-REASSIGNMENT IN forward(): Never write `self.<buffer_or_param> = ...` inside `forward()`. Use a local variable.
- NUMERICAL SAFETY: The argument to `torch.exp()` must be negative or clamped from above. Use `.clamp(min=1e-15)` before `.sqrt()` or `.log()`.
- FORWARD SIGNATURE: Always `def forward(self, x1, x2, diag=False, **params)`.
- DEVICE: Use `register_buffer(...)` for tensors created in `__init__`, or call `.to(x1.device)` in `forward()`. Any new tensor created inside `forward()` must be on `x1.device`.
- IMPORTS: Always include `import math` if using math.sqrt/math.log.
- DIAG: When `diag=True`, return shape (..., N): `if diag: return covar.diagonal(dim1=-2, dim2=-1)`.
- NO IN-PLACE OPERATIONS: Never use `+=`, `*=`, `.add_()`, `.mul_()`, `.clamp_()`, `.pow_()`, `.exp_()`, `.sqrt_()`. In-place ops break the autograd graph.
- NO DETACH/NUMPY IN forward(): Never call `.detach()`, `.numpy()`, `np.array()`, `.item()`, or use `torch.no_grad()` inside `forward()`. These break the autograd graph.
Return ONLY valid Python code in a ```python block."""

    try:
        text = client.generate(prompt, max_tokens=8192)
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1) if match else text
    except Exception as e:
        print(f"    [Generation] Error: {e}")
        return population_subset[0]["code"]


def multi_kernel_composition(population_subset, client: LLMClient | None = None, max_compositions=3):
    if client is None:
        return population_subset[0]["code"]
    return _multi_kernel_composition_impl(population_subset, client, max_compositions=max_compositions)


def multi_kernel_composition_parallel(temp_pool, client: LLMClient | None = None, n_compositions=4, max_compositions=3, simplicity_prompt=False):
    if client is None or len(temp_pool) < 2:
        return []
    pool_size = min(len(temp_pool), 5)
    subset = temp_pool[:pool_size]

    prompt = f"Top {len(subset)} EvolvedKernel classes available to compose:\n\n"
    for i, ind in enumerate(subset):
        prompt += f"--- Rank {i+1} Kernel (NLL Loss: {ind.get('loss', float('inf')):.4f}) ---\n```python\n{ind['code']}\n```\n\n"
        
    prompt += f"""Your task is to generate {n_compositions} DISTINCT, DIVERSE, and highly innovative composed GP kernels. 
For EACH of the {n_compositions} combinations:
- YOU decide which kernels to combine: choose k (2 <= k <= {len(subset)}) from the above. Pick the ones you think will work best together.
- Combine using Addition or Multiplication or other grouping structures.
- LIMIT the total number of combination operators (like + or *) to AT MOST {max_compositions}.
- Mathematically explore fundamentally distinct combinations. DO NOT just output the same combination {n_compositions} times.

CRITICAL RULES:
1. CLASS NAME: The class MUST be named exactly `EvolvedKernel`.
2. LENGTHSCALE: Choose ONE approach. APPROACH A (recommended): set `has_lengthscale = True` and pass `lengthscale_prior`/`lengthscale_constraint` to `super().__init__()` — do NOT also call `register_parameter("raw_lengthscale", ...)`. APPROACH B: do NOT set `has_lengthscale = True`, manually register `raw_lengthscale` and define a `@property`. NEVER write `self.lengthscale = <anything>` — raises `KeyError: "attribute 'lengthscale' already exists"`.
3. PARAMETER REGISTRATION ORDER: Always call `register_parameter(name, ...)` BEFORE `register_constraint(name, ...)` for the same parameter.
4. SHAPE INVARIANT: `forward()` MUST return exactly `(..., N1, N2)` for ANY N1, N2 (e.g., N1=3, N2=7 non-square). `x1` is `(..., N1, D)`, `x2` is `(..., N2, D)`. Only reference `self.attribute` values defined in `__init__`. Never materialize `(..., N1, N2, D)` intermediate; use `torch.cdist(x1s, x2s, p=2)` or `x1 @ x2.transpose(-1,-2)`. Mentally verify output shape with N1=3, N2=7.
5. NO SELF-REASSIGNMENT IN forward(): Never write `self.<buffer_or_param> = ...` inside `forward()`. Use a local variable.
6. NUMERICAL SAFETY: The argument to `torch.exp()` must be negative or clamped from above. Use `.clamp(min=1e-15)` before `.sqrt()` or `.log()`.
7. DIAG: When `diag=True`, return shape (..., N): `if diag: return covar.diagonal(dim1=-2, dim2=-1)`.
8. NO IN-PLACE OPERATIONS: Never use `+=`, `*=`, `.add_()`, `.mul_()`, `.clamp_()`, `.pow_()`, `.exp_()`, `.sqrt_()`. In-place ops break the autograd graph.
9. NO DETACH/NUMPY IN forward(): Never call `.detach()`, `.numpy()`, `np.array()`, `.item()`, or use `torch.no_grad()` inside `forward()`. These break the autograd graph and cause gradient failures during acquisition optimization. Any new tensor created inside `forward()` must be on `x1.device`.

OUTPUT FORMAT:
Generate EXACTLY {n_compositions} distinct python code blocks, each enclosed in ```python ... ```. Each must contain a full self-contained implementation of `class EvolvedKernel(gpytorch.kernels.Kernel):`. Do not write any explanations.
"""
    if simplicity_prompt:
        prompt += """
SIMPLICITY CONSTRAINT (IMPORTANT):
- Prefer simple compositions: a clean sum (k1 + k2) or product (k1 * k2) of 2 kernels.
- Avoid deeply nested or multi-layer compositions with many operations.
- A simple k1 + k2 often outperforms a complex 5-operation kernel in high dimensions.
- Fewer parameters = faster fitting = better generalization. Keep it minimal.
"""
    try:
        text = client.generate(prompt, temperature=1.0, max_tokens=8192)
        matches = re.finditer(r"```python\n(.*?)\n```", text, re.DOTALL)
        codes = [m.group(1).strip() for m in matches]
    except Exception as e:
        print(f"    [Generation] Error generating joint compositions: {e}")
        codes = []

    if not codes:
        codes = [subset[0]["code"]]
    while len(codes) < n_compositions:
        codes.append(codes[-1])
        
    return [(c, pool_size) for c in codes[:n_compositions]]


def mutate_codes_parallel(parent_pairs, client: LLMClient | None = None, max_workers=None, max_compositions=3, n_mutations=5, temperature=1.0):
    if not parent_pairs:
        return []
    if client is None:
        return [parent_pairs[0][0]] * n_mutations

    context_str = ""
    for i, (code, loss, eval_time) in enumerate(parent_pairs):
        time_info = f", Eval Time: {eval_time:.6f}s" if eval_time else ""
        context_str += f"\n--- Rank {i+1} Kernel ---\nNLL Loss: {loss:.4f}{time_info}\n```python\n{code}\n```\n"

    prompt = f"""You are an expert for design GP kernel suitable to high-dimensional Bayesian optimization.
I will provide you with the TOP {len(parent_pairs)} current best performing kernels in our population.
Your goal is to generate {n_mutations} UNSEEN, CREATIVE, DISTINCT, mathematically valid, and highly innovative mutated versions of these kernels.
{context_str}

CRITICAL DISCOVERY INSTRUCTIONS:
1. HIGH-DIMENSIONAL ROBUSTNESS: In higher dimensions, standard kernels fail due to boundary-seeking behavior and distance collapse. Here are some examples but not limited to:
   - Spherical Projections: Map high-dimensional inputs to a hypersphere using transformations like the inverse stereographic projection.
   - Cylindrical Projections: Map the search space to a hyper-cylinder before applying the distance metric. Unlike spherical projections, this is often highly effective when paired with high-capacity base kernels (like RBF or Matérn) rather than simple linear ones.
   - Dimensionality Scaling: Scale inputs or global lengthscales by $O(\sqrt{{D}})$ (e.g., math.sqrt(3.0 / D)) so distance metrics do not collapse as $D \to \infty$.
   - Simplex Constraints: For polynomial or combined kernels, enforce that coefficients sum to 1 using torch.nn.functional.softmax to remove implicit outputscales.
2. STRUCTURAL ASSUMPTIONS: You are highly encouraged to build kernels that impose structural assumptions on the high-dimensional space. Here are some examples but not limited to:
    - Subspace Embeddings: Project inputs into a lower-dimensional latent space via learnable linear weights (e.g., x_latent=xW) before computing distances.
    - Additive Decompositions: Split the D dimensions into disjoint subsets (or groups), compute the kernel independently on each subset, and sum the results.
    - Decoupled Lengthscales: Separate the lengthscale into a single global scalar (scaled by sqrt(D)) multiplied by an unconstrained ARD vector representing dimension-specific sensitivities.
    - Variable Selection: Introduce learnable sparse attention weights or gating mechanisms across the D dimensions to suppress irrelevant features.
3. COMPOSITIONS: You can combine multiple kernels using addition or multiplication. 
4. You must write a class named `EvolvedKernel` that inherits from `gpytorch.kernels.Kernel`.
5. Keep code CONCISE: avoid verbose comments, redundant docstrings, or unnecessary boilerplate. Shorter mutations are preferred; long code is penalized in selection.
6. Keep computation EFFICIENT: use optimized vectorized operations. High time complexity is penalized in selection.

COMMON PITFALLS TO AVOID:
1. LENGTHSCALE MANAGEMENT — You MUST choose ONE of these two approaches, NEVER mix them:
   APPROACH A (RECOMMENDED): Set `has_lengthscale = True` and pass `lengthscale_prior`/`lengthscale_constraint` to `super().__init__()`. GPyTorch will automatically create `self.raw_lengthscale`, `self.lengthscale`, and the constraint. Do NOT also call `self.register_parameter("raw_lengthscale", ...)` — this creates a duplicate/conflict that causes crashes.
   APPROACH B: Do NOT set `has_lengthscale = True`. Manually register `self.register_parameter("raw_lengthscale", ...)`, `self.register_constraint(...)`, and define a `@property` for `lengthscale` that applies the constraint transform. Do NOT pass `lengthscale_prior`/`lengthscale_constraint` to `super().__init__()`.
   NEVER assign the return value of `register_parameter()` to `self.lengthscale` — `register_parameter` returns `None`.
   NEVER write `self.lengthscale = <anything>` (or `self.raw_lengthscale = <anything>`) in any form. The base `gpytorch.kernels.Kernel` exposes `lengthscale` as a property descriptor, and direct assignment raises `KeyError: "attribute 'lengthscale' already exists"` at instantiation. If you want a custom length-scale-like tensor, pick a DIFFERENT name (e.g. `raw_inv_scale`, `log_ls`).
2. PARAMETER REGISTRATION ORDER — Always call `register_parameter(name, ...)` BEFORE `register_constraint(name, ...)` for the same parameter name. Calling `register_constraint` first raises `"Attempting to register constraint for nonexistent parameter"`.
3. SHAPE INVARIANT — `forward()` MUST return exactly `(..., N1, N2)` for ANY combination of N1 and N2, including non-square (e.g., N1=3, N2=7). `x1` has shape `(..., N1, D)` and `x2` has shape `(..., N2, D)`. Only reference `self.attribute` values that are explicitly defined in `__init__` — referencing undefined attributes causes `AttributeError` in `forward()`. Do NOT attach trailing dims with `[..., None]`, `.unsqueeze(-1)`, or `.view(1, -1, 1)` to per-coefficient weights — reduce to scalar first. Never materialize `(..., N1, N2, D)` intermediate (reduce D via `torch.cdist` or `matmul`). Mentally verify output shape with N1=3, N2=7.
4. NO SELF-REASSIGNMENT IN forward() — Never write `self.<buffer_or_param> = ...` inside `forward()` (e.g. `self.inv_lengthscale = 1 / self.lengthscale`). This mutates a registered tensor, breaks the autograd graph, and causes fit failures. Use a LOCAL variable instead: `inv_ls = 1.0 / self.lengthscale`.
5. NUMERICAL SAFETY — The argument to `torch.exp()` must be negative (or clamped from above). Never write `torch.exp(+c * x.pow(2).sum(...))` — positive quadratics in the exponent diverge in high dimensions and crash fitting.
6. PAIRWISE DISTANCE IN forward() — Use `torch.cdist(x1s, x2s, p=2)` (with `x1s = x1 / self.lengthscale`) for Euclidean / squared distances → shape `(..., N, M)`. For inner-product kernels use `torch.matmul(x1, x2.transpose(-1, -2))`. Both reduce D without building a `(..., N, M, D)` intermediate.
7. IMPORT ALL USED MODULES — Always include `import math` if using `math.sqrt`, `math.log`, `math.pi`, etc. Missing imports cause immediate failure at exec() time.
8. DO NOT USE NON-EXISTENT GPYTORCH CLASSES — Only use classes that actually exist in gpytorch.
9. DIAG HANDLING — When `diag=True`, return a 1-D tensor of shape (..., N), NOT a matrix. Use: `if diag: return covar.diagonal(dim1=-2, dim2=-1)`.
10. NUMERICAL STABILITY — Clamp denominators and arguments to log/sqrt: use `.clamp(min=1e-15)` before `.sqrt()` or `.log()`.
11. NO IN-PLACE OPERATIONS — Use `x = x + y` instead of `x += y`. Use `x = x.clamp(...)` instead of `x.clamp_(...)`. Never use `.add_()`, `.mul_()`, `.pow_()`, `.exp_()`, `.sqrt_()`. In-place operations break the autograd graph.
12. NO DETACH/NUMPY IN forward() — Never call `.detach()`, `.numpy()`, `np.array()`, `.item()`, or use `torch.no_grad()` inside `forward()`. These break the autograd graph and cause silent gradient failures during acquisition optimization. Any new tensor created inside `forward()` must be on `x1.device` (e.g., `torch.zeros(..., device=x1.device)`).

OUTPUT FORMAT:
Generate EXACTLY {n_mutations} distinct python code blocks, each enclosed in ```python ... ```. Each must contain a full self-contained implementation of `class EvolvedKernel(gpytorch.kernels.Kernel):`. Do not write anything else.
DO NOT REPEAT the same structure with minor tweaks. Each code block MUST reflect a fundamentally distinct evolutionary path.
"""
    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=8192)
        matches = re.finditer(r"```python\n(.*?)\n```", text, re.DOTALL)
        codes = [m.group(1).strip() for m in matches]
    except Exception as e:
        print(f"    [Generation] Error generating joint mutations: {e}")
        codes = []

    if not codes:
        codes = [parent_pairs[0][0]]
    while len(codes) < n_mutations:
        codes.append(codes[-1])

    return codes[:n_mutations]


BASE_KERNEL_PRESETS = {
    "default": (
        ["dsp", "matern52", "spherical", "cylindrical", "rq"],
        [DSP_KERNEL_CODE, MATERN52_KERNEL_CODE, SPHERICAL_KERNEL_CODE, CYLINDRICAL_KERNEL_CODE, RQ_KERNEL_CODE],
    ),
    "standard": (
        ["dsp", "matern32", "matern52", "rq", "linear", "periodic"],
        [DSP_KERNEL_CODE, MATERN32_KERNEL_CODE, MATERN52_KERNEL_CODE, RQ_KERNEL_CODE, LINEAR_KERNEL_CODE, PERIODIC_KERNEL_CODE],
    ),
}

BASE_KERNEL_NAMES, BASE_KERNEL_CODES = BASE_KERNEL_PRESETS["default"]

BASE_KERNELS = BASE_KERNEL_NAMES
DISCOVERED_KERNELS = ["discovery_1", "discovery_2"]
DISCOVERED_KERNEL_CODES = [DISCOVERY_1_KERNEL_CODE, DISCOVERY_2_KERNEL_CODE]


class MLEBackend:
    client: LLMClient | None = None

    def get_base_codes(self, input_dim=None, preset="default"):
        base_names, base_codes = BASE_KERNEL_PRESETS.get(preset, BASE_KERNEL_PRESETS["default"])
        return base_names, base_codes

    def evaluate_kernel_code(self, code, train_x, train_y, model_selection="loss", state_dict=None, skip_fit=False, timeout=120.0, bounds=None, strict_load=True, eval_subsample=1.0, complexity_lambda=0.0):
        return evaluate_kernel_code(code, train_x, train_y, model_selection=model_selection, state_dict=state_dict, skip_fit=skip_fit, timeout=timeout, bounds=bounds, strict_load=strict_load, eval_subsample=eval_subsample, complexity_lambda=complexity_lambda)

    def mutate_code_with_llm(self, code, loss, eval_time, client, max_compositions=3, temperature=1.0):
        return mutate_code_with_llm(code, loss, current_time=eval_time, client=client, max_compositions=max_compositions, temperature=temperature)

    def fix_code_with_llm(self, code, fail_reason, tb_str, client, temperature=0.7):
        return fix_code_with_llm(code, fail_reason, tb_str, client=client, temperature=temperature)

    def mutate_codes_parallel(self, parent_pairs, client, max_workers=None, max_compositions=3, n_mutations=5, temperature=1.0):
        return mutate_codes_parallel(parent_pairs, client, max_workers, max_compositions=max_compositions, n_mutations=n_mutations, temperature=temperature)

    def multi_kernel_composition(self, subset, client, max_compositions=3):
        return multi_kernel_composition(subset, client, max_compositions=max_compositions)

    def multi_kernel_composition_parallel(self, temp_pool, client, n_compositions=4, max_compositions=3):
        return multi_kernel_composition_parallel(temp_pool, client, n_compositions, max_compositions=max_compositions)

