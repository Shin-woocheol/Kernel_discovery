"""
Formula-level kernel evolution: LLM mutation/composition + code generation.

Three-stage pipeline:
  1. Mutation/Composition at the mathematical formula level (LLM)
  2. Code generation from formula (LLM)
  3. Validation (fast) then fitting (heavy, parallelized)

Prompts, parsers, base kernel formulas, and validate/fit functions live here.
"""
import math
import re
import time
import traceback as _traceback

import torch
import gpytorch

from .llm_client import LLMClient
from .bo import _kernel_from_code, _initialize_model, _fit_model
from .loocv import loocv_nlpd_loss, loocv_crps_loss, rank_weighted_loss


# ---------------------------------------------------------------------------
# Base kernel formulas (hardcoded, matching bo.py base kernel codes)
# ---------------------------------------------------------------------------

BASE_FORMULAS = {
    "dsp_gaussian": """\
KERNEL: DSP Gaussian (Squared Exponential with Decoupled Scaled Prior)

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale with LogNormal prior, loc=sqrt(2)+0.5*log(1), scale=sqrt(3)

INPUT TRANSFORM:
  x_scaled = x / lengthscale

COVARIANCE FUNCTION:
  r² = ||x_scaled1 - x_scaled2||²
  k(x1, x2) = exp(-r²/2)

PSD GUARANTEE: Standard RBF (squared exponential) kernel — always PSD.""",

    "matern52": """\
KERNEL: Matérn-5/2

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale with LogNormal prior, loc=sqrt(2)+0.5*log(1), scale=sqrt(3)

INPUT TRANSFORM:
  x_scaled = x / lengthscale

COVARIANCE FUNCTION:
  r = ||x_scaled1 - x_scaled2||   (clamped ≥ 1e-15)
  k(x1, x2) = (1 + √5·r + 5/3·r²) · exp(-√5·r)

PSD GUARANTEE: Matérn kernel with ν=5/2 is a standard PSD kernel.""",

    "rq": """\
KERNEL: Rational Quadratic

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale with LogNormal prior
- alpha (scalar, positive): mixture scale parameter

INPUT TRANSFORM:
  x_scaled = x / lengthscale

COVARIANCE FUNCTION:
  r² = ||x_scaled1 - x_scaled2||²
  k(x1, x2) = (1 + r²/(2·alpha))^(-alpha)

PSD GUARANTEE: RQ kernel is an infinite mixture of RBF kernels with different lengthscales — always PSD.""",

    "spherical": """\
KERNEL: Spherical Polynomial (Inverse Stereographic Projection)

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale
- global_ls (scalar, sigmoid → [0,1]): global lengthscale factor
- poly_coeffs ((2,), none → softmax): polynomial mixing coefficients

INPUT TRANSFORM:
  x_centered = (x - center) / lengthscale
  max_sq_norm = ||(max - min) / (2·lengthscale)||²
  x_scaled = x_centered / sqrt(global_ls · max_sq_norm)
  x_sphere = inv_stereographic(x_scaled)
           = [2·x_scaled, ||x_scaled||²-1] / (||x_scaled||²+1)

COVARIANCE FUNCTION:
  c = softmax(poly_coeffs)           # simplex constraint
  φ(x) = [x_sphere · √c[1], √c[0]]  # augmented feature vector
  k(x1, x2) = φ(x1)ᵀ · φ(x2)        # linear kernel on augmented sphere

PSD GUARANTEE: k = φ(x1)ᵀφ(x2) for explicit feature map φ — always PSD.""",

    "cylindrical": """\
KERNEL: Cylindrical (Radial × Angular decomposition)

PARAMETERS:
- angular_weights ((4,), positive): polynomial weights for angular kernel
- alpha (scalar, positive): Kumaraswamy shape parameter a
- beta (scalar, positive): Kumaraswamy shape parameter b
- radial_base_kernel: Matérn-5/2 kernel applied to warped radii

INPUT TRANSFORM:
  x_normalized = (x - center) / radius       # map to unit ball
  r = ||x_normalized||                        # radial component ∈ [0,1]
  a = x_normalized / r                        # angular direction (unit vector)
  r_warped = Kumaraswamy_CDF(r; alpha, beta)  # warp radius: 1-(1-r^α)^β

COVARIANCE FUNCTION:
  angular_kernel(a1, a2) = Σ_{p=0}^{3} w_p · (a1ᵀa2)^p    # polynomial on dot product
  radial_kernel(r1, r2) = Matérn52(r_warped1, r_warped2)
  k(x1, x2) = radial_kernel(r1, r2) · angular_kernel(a1, a2)

PSD GUARANTEE: Product of two PSD kernels (radial Matérn and angular polynomial with positive weights) is PSD.""",

    "dsp": """\
KERNEL: DSP Gaussian (Squared Exponential with Decoupled Scaled Prior)

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale with LogNormal prior, loc=sqrt(2)+0.5*log(1), scale=sqrt(3)

INPUT TRANSFORM:
  x_scaled = x / lengthscale

COVARIANCE FUNCTION:
  r² = ||x_scaled1 - x_scaled2||²
  k(x1, x2) = exp(-r²/2)

PSD GUARANTEE: Standard RBF (squared exponential) kernel — always PSD.""",

    "matern32": """\
KERNEL: Matérn-3/2

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale with LogNormal prior, loc=sqrt(2)+0.5*log(1), scale=sqrt(3)

INPUT TRANSFORM:
  x_scaled = x / lengthscale

COVARIANCE FUNCTION:
  r = ||x_scaled1 - x_scaled2||   (clamped ≥ 1e-15)
  k(x1, x2) = (1 + √3·r) · exp(-√3·r)

PSD GUARANTEE: Matérn kernel with ν=3/2 is a standard PSD kernel.""",

    "linear": """\
KERNEL: Linear (dot-product)

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale
- offset (scalar, positive): bias term

INPUT TRANSFORM:
  x_scaled = x / lengthscale

COVARIANCE FUNCTION:
  k(x1, x2) = x_scaled1ᵀ · x_scaled2 + offset

PSD GUARANTEE: k = φ(x1)ᵀφ(x2) for feature map φ(x)=[x_scaled, √offset] — always PSD.""",

    "periodic": """\
KERNEL: Periodic (standard sine-based)

PARAMETERS:
- lengthscale (per_dim, positive): ARD lengthscale controlling smoothness within each period
- period (per_dim, positive): period length per dimension

COVARIANCE FUNCTION:
  s_d = sin(π · |x1_d - x2_d| / period_d) / lengthscale_d    (per dimension d)
  r² = Σ_d s_d²
  k(x1, x2) = exp(-2 · r²)

PSD GUARANTEE: exp(-2·||sin(π·Δ/p)/l||²) is PSD as a product of per-dimension PSD periodic kernels.""",
}


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_formula_blocks(text: str) -> list[str]:
    """Extract formula blocks from LLM output.

    Primary: ```formula ... ``` fenced blocks (same pattern as v1's ```python).
    Fallback 1: split on '---' separator.
    Fallback 2: return entire text as single formula.
    """
    # --- Primary: fenced formula blocks ---
    matches = re.findall(r"```formula\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    # --- Fallback 1: --- separator ---
    parts = re.split(r"\n---+\n", text)
    # Filter out short noise (< 30 chars likely not a real formula)
    parts = [p.strip() for p in parts if len(p.strip()) > 30]
    if len(parts) > 1:
        return parts

    # --- Fallback 2: whole text ---
    return [text.strip()] if text.strip() else []


def parse_code_block(text: str) -> str | None:
    """Extract first ```python ... ``` block from LLM output."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _score_label(model_selection: str) -> tuple[str, str]:
    """Return (score_name, direction) for prompt context."""
    if model_selection in ("crps", "loocv_crps"):
        return "CRPS", "lower is better"
    elif model_selection in ("nlpd", "loocv_nlpd"):
        return "NLPD", "lower is better"
    elif model_selection in ("bic", "bic_correct"):
        return "BIC", "lower is better"
    elif model_selection == "crps_bic":
        return "CRPS-BIC", "lower is better"
    return "NLL", "lower is better"


def is_composition_output(p: dict) -> bool:
    """True if p was produced by a composition stage (unsuitable as a composition parent)."""
    if p.get("stage") == "composition":
        return True
    src = p.get("source", "")
    return "composition_gen" in src or "_comp_fallback" in src


_SIMPLICITY_BLOCK = """
SIMPLICITY PREFERENCE (IMPORTANT — complex kernels rarely outperform simple ones):
- Prefer kernels with at most D+5 total learnable scalar parameters (D = input dimension).
- Avoid nested nonlinearities (exp(exp(...)), tanh(log(...))).
- A clean RBF/Matérn with one input projection often beats exotic multi-parameter kernels.
- Fewer parameters = faster fitting = better generalization in high dimensions.
- Do NOT use (D, D) projection matrices — they explode with input dimension.
"""


_HIGHDIM_BLOCK = """
HIGH-DIMENSIONAL STRATEGIES (examples, not limited to):
1. Input projections: stereographic, cylindrical, tanh warping, linear embedding to lower-dim subspace
2. Dimensionality scaling: scale distances by sqrt(D) to prevent distance collapse
3. Structural assumptions: additive decompositions, subspace embeddings, sparse attention/gating
4. Novel distance metrics: geodesic, learned metrics, projected distances
5. Non-stationarity: input-dependent lengthscales, warping functions
6. Kernel arithmetic: sum (smoothness mixing) or product (interaction capture) of PSD terms
"""


_PSD_BLOCK_MUTATION = """
POSITIVE SEMI-DEFINITENESS (CRITICAL — GP requires PSD kernels):
- Safe building blocks: RBF exp(-r²), Matérn, RQ, linear k=xᵀy, polynomial (xᵀy+c)^p with positive coefficients
- Safe operations: sum of PSD kernels, product of PSD kernels, φ(x)ᵀφ(x') for any feature map φ
- Risky: custom formulas not built from these — the system will reject non-PSD kernels
"""


_PSD_BLOCK_COMPOSITION = """
POSITIVE SEMI-DEFINITENESS (CRITICAL):
- k1 + k2 is PSD if both are PSD
- k1 * k2 is PSD if both are PSD (Schur product theorem)
- Ensure your composition preserves PSD
"""


def build_mutation_prompt(
    parent_formulas_with_loss: list[tuple[str, float]],
    n_mutations: int,
    model_selection: str = "loss",
    simplicity_guidance: bool = False,
    include_highdim: bool = True,
    include_psd: bool = True,
) -> str:
    score_name, direction = _score_label(model_selection)

    context = ""
    for i, (formula, loss) in enumerate(parent_formulas_with_loss):
        context += f"\n--- Rank {i+1} ({score_name}: {loss:.4f}, {direction}) ---\n{formula}\n"

    simplicity = _SIMPLICITY_BLOCK if simplicity_guidance else ""
    highdim = _HIGHDIM_BLOCK if include_highdim else ""
    psd = _PSD_BLOCK_MUTATION if include_psd else ""

    return f"""\
You are an expert in Gaussian process kernel design for high-dimensional Bayesian optimization.

TOP {len(parent_formulas_with_loss)} kernels in our population:
{context}

Generate {n_mutations} DISTINCT, CREATIVE, mathematically innovative MUTATED kernels.
Each mutation should be derived from one or more of the above kernels, but with meaningful mathematical changes.
{highdim}{psd}{simplicity}
OUTPUT FORMAT:
Return EXACTLY {n_mutations} kernel descriptions. Enclose EACH in a ```formula block:

```formula
KERNEL: [name]

PARAMETERS:
- [name] ([shape], [constraint]): [description]

INPUT TRANSFORM:
  [step-by-step math]

COVARIANCE FUNCTION:
  [k(x1, x2) = ...]

PSD GUARANTEE:
  [why this kernel is PSD]
```

Each MUST be fundamentally distinct (different transform, different covariance structure, different mathematical idea)."""


def build_composition_prompt(
    population_formulas_with_loss: list[tuple[str, float]],
    n_compositions: int,
    model_selection: str = "loss",
    simplicity_guidance: bool = False,
    include_psd: bool = True,
) -> str:
    score_name, direction = _score_label(model_selection)

    context = ""
    for i, (formula, loss) in enumerate(population_formulas_with_loss):
        label = chr(ord("A") + i) if i < 26 else str(i)
        context += f"\n--- Kernel {label} ({score_name}: {loss:.4f}, {direction}) ---\n{formula}\n"

    simplicity = _SIMPLICITY_BLOCK if simplicity_guidance else ""
    psd = _PSD_BLOCK_COMPOSITION if include_psd else ""

    return f"""\
You are combining GP kernels for high-dimensional Bayesian optimization.

Population of {len(population_formulas_with_loss)} kernels:
{context}

Generate {n_compositions} DISTINCT composed kernels. For EACH composition:
- Pick exactly 2 kernels from the population that COMPLEMENT each other mathematically
- Combine them: sum their covariance functions, multiply them, share a transform with different distance metrics, or any other valid composition
- Explain WHY these two kernels complement each other
{psd}{simplicity}
OUTPUT FORMAT:
Return EXACTLY {n_compositions} kernel descriptions. Enclose EACH in a ```formula block:

```formula
KERNEL: [name]
COMPOSED FROM: [Kernel X] + [Kernel Y]  (or *, or other operation)
WHY: [1 sentence on why these complement each other]

PARAMETERS:
- [param] ([shape], [constraint]): [description]

INPUT TRANSFORM:
  [step-by-step math]

COVARIANCE FUNCTION:
  k(x1, x2) = ...

PSD GUARANTEE:
  [why this kernel is PSD]
```"""


def build_codegen_prompt(formula: str) -> str:
    return f"""\
You are a GPyTorch kernel code generator. Convert this kernel formula into a GPyTorch EvolvedKernel class.

{formula}

REFERENCE TEMPLATE A — Distance-based kernel (RBF, Matérn, RQ):
```python
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, *, ard_num_dims: int,
                 bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
                 batch_shape: torch.Size = torch.Size([]), **kwargs):
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0))
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode)
        super().__init__(
            ard_num_dims=ard_num_dims, batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint)
        self.D = ard_num_dims

    def forward(self, x1, x2, diag=False, **params):
        # x1: (..., N1, D), x2: (..., N2, D) — may have batch dims!
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist = torch.cdist(x1s, x2s, p=2)              # (..., N1, N2)
        covar = torch.exp(-0.5 * dist.pow(2))
        if diag:
            return covar.diagonal(dim1=-2, dim2=-1)
        return covar
```

REFERENCE TEMPLATE B — Feature-map kernel (spherical projection, polynomial):
```python
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, *, ard_num_dims: int,
                 bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
                 batch_shape: torch.Size = torch.Size([]), **kwargs):
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0))
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode)
        super().__init__(
            ard_num_dims=ard_num_dims, batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint)
        self.D = ard_num_dims
        # Polynomial mixing coefficients
        self.register_parameter("raw_coeffs", torch.nn.Parameter(torch.zeros(2)))

    def _project_to_sphere(self, x):
        # x: (..., N, D) → sphere: (..., N, D+1)
        x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)           # (..., N, 1)
        denom = (x_sq_norm + 1.0).clamp(min=1e-15)
        return torch.cat([2 * x, x_sq_norm - 1.0], dim=-1) / denom  # (..., N, D+1)

    def forward(self, x1, x2, diag=False, **params):
        # x1: (..., N1, D), x2: (..., N2, D) — may have leading batch dims!
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        p1 = self._project_to_sphere(x1s)                        # (..., N1, D+1)
        p2 = self._project_to_sphere(x2s)                        # (..., N2, D+1)
        coeffs = torch.nn.functional.softmax(self.raw_coeffs, dim=0)

        # Build augmented features: (..., N, D+2)
        # Use [... :1] slicing for batch-safe scalar expansion — NEVER .expand(x.size(0), 1)
        phi1 = torch.cat([p1 * coeffs[1].sqrt(), coeffs[0].sqrt() * torch.ones_like(p1[..., :1])], dim=-1)
        phi2 = torch.cat([p2 * coeffs[1].sqrt(), coeffs[0].sqrt() * torch.ones_like(p2[..., :1])], dim=-1)

        covar = phi1 @ phi2.transpose(-1, -2)                    # (..., N1, N2)
        if diag:
            return covar.diagonal(dim1=-2, dim2=-1)
        return covar
```

CRITICAL RULES:
1. __init__ SIGNATURE: `__init__` MUST accept `ard_num_dims: int` as the ONLY required parameter. The system calls `EvolvedKernel(ard_num_dims=D)`. All other args MUST have defaults. NEVER add required args like `q: int`, `center: Tensor`, `M: int`.
2. FORWARD SIGNATURE: `def forward(self, x1, x2, diag=False, **params)` — always accept **params.
3. BATCH-SAFE SHAPES (CRITICAL): The kernel is tested with these exact shapes — your code MUST handle ALL of them:
   - 2D: x1=(5, D) vs x2=(1, D) → output must be (5, 1), NOT (5, 5)
   - 2D: x1=(3, D) vs x2=(7, D) → output must be (3, 7)
   - 3D batched (BoTorch acqf optimization): x1=(1, 4, D) vs x2=(1, 3, D) → output must be (1, 4, 3)
   Therefore: ALL dim indexing must be relative (-1 for D, -2 for N). NEVER use `.size(0)`, `.expand(x.size(0), ...)`, or `.shape[0]`. For broadcasting scalars to match batch+N dims, use `torch.ones_like(x[..., :1])`.
4. OUTPUT SHAPE: return (..., N1, N2) for ANY N1 != N2. If your output is (N1, N1) instead of (N1, N2), you have a bug.
5. DISTANCE: `torch.cdist(x1s, x2s, p=2)` — NEVER `x1.unsqueeze(-2) - x2.unsqueeze(-3)` (OOM).
6. INNER PRODUCT: `x1 @ x2.transpose(-1, -2)`.
7. DEVICE: new tensors in forward use `device=x1.device`. Use `register_buffer` in __init__.
8. NO IN-PLACE: `x = x + y` not `x += y`. `x = x.clamp(...)` not `x.clamp_(...)`.
9. NO DETACH/NUMPY: never `.detach()`, `.numpy()`, `.item()`, `torch.no_grad()` in forward.
10. NUMERICAL: `.clamp(min=1e-15)` before sqrt/log, `.clamp(max=20.0)` for exp.
11. LENGTHSCALE: if `has_lengthscale=True`, do NOT `register_parameter("raw_lengthscale", ...)`.
12. PARAM ORDER: `register_parameter(name, ...)` BEFORE `register_constraint(name, ...)`.
13. DIAG: `if diag: return covar.diagonal(dim1=-2, dim2=-1)`.
14. IMPORTS: include `import math` if using math.sqrt/pi/log.
15. NO SELF-REASSIGNMENT in forward(): never `self.x = ...` — use local variables.
16. ONLY USE EXISTING gpytorch classes.
17. NO SUB-KERNELS: do NOT instantiate gpytorch kernel objects (e.g. `MaternKernel()`, `RBFKernel()`) as sub-components. They produce LazyEvaluatedKernelTensor with unpredictable shapes. Instead, implement the formula directly (e.g. for Matern-5/2: `(1 + sqrt5*d + 5/3*d**2) * exp(-sqrt5*d)`).
18. NO DATA-DEPENDENT BOUNDS in forward(): never compute bounds/center from the input data (e.g. `x1.max(dim=-2)` or `x1.min(dim=-2)`). These change with N and break shape invariance. Store fixed bounds via `register_buffer` in __init__, or use constants (e.g. 0 and 1).
19. SCALAR PAIRWISE DISTANCE: when computing distance between scalar values (e.g. warped radii of shape (..., N, 1)), use `torch.cdist(r1, r2, p=2)` which gives (..., N1, N2). NEVER use `r1 - r2` or `torch.abs(r1 - r2)` — this gives (..., N1, 1) not (..., N1, N2) and breaks cross-pair computation.

Return ONLY the Python code in a ```python block."""


def build_fix_formula_prompt(formula: str, error_msg: str) -> str:
    return f"""\
This kernel formula produced code that failed evaluation:

{formula}

Error: {error_msg}

Fix the FORMULA (not the code — code will be regenerated from the fixed formula).
The most common issue is non-PSD covariance function. Ensure:
- All terms are known-PSD building blocks (RBF, Matérn, RQ, linear, polynomial)
- Combined only via sum or product
- No custom formulas that aren't provably PSD

Return ONLY the fixed formula in a ```formula block."""


# ---------------------------------------------------------------------------
# LLM call wrappers
# ---------------------------------------------------------------------------

def generate_formulas_mutation(
    parent_formulas_with_loss: list[tuple[str, float]],
    n_mutations: int,
    client: LLMClient,
    temperature: float = 1.0,
    model_selection: str = "loss",
    simplicity_prompt: bool = False,
    include_highdim: bool = True,
    include_psd: bool = True,
) -> list[str]:
    """Call LLM to generate n_mutations mutated formulas in a single call."""
    prompt = build_mutation_prompt(
        parent_formulas_with_loss, n_mutations, model_selection, simplicity_prompt,
        include_highdim=include_highdim, include_psd=include_psd,
    )
    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=8192)
    except Exception as e:
        print(f"    [Mutation] LLM error: {e}")
        return []
    formulas = parse_formula_blocks(text)
    return formulas[:n_mutations]


def generate_formulas_composition(
    population_formulas_with_loss: list[tuple[str, float]],
    n_compositions: int,
    client: LLMClient,
    temperature: float = 1.0,
    model_selection: str = "loss",
    simplicity_prompt: bool = False,
    include_psd: bool = True,
) -> list[str]:
    """Call LLM to generate n_compositions composed formulas in a single call."""
    prompt = build_composition_prompt(
        population_formulas_with_loss, n_compositions, model_selection, simplicity_prompt,
        include_psd=include_psd,
    )
    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=8192)
    except Exception as e:
        print(f"    [Composition] LLM error: {e}")
        return []
    formulas = parse_formula_blocks(text)
    return formulas[:n_compositions]


def generate_code_from_formula(
    formula: str,
    client: LLMClient,
    temperature: float = 0.3,
) -> str | None:
    """Call LLM to convert a formula into GPyTorch EvolvedKernel code."""
    prompt = build_codegen_prompt(formula)
    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=8192)
    except Exception as e:
        print(f"    [CodeGen] LLM error: {e}")
        return None
    return parse_code_block(text)


def fix_formula(
    formula: str,
    error_msg: str,
    client: LLMClient,
    temperature: float = 0.7,
) -> str | None:
    """Call LLM to fix a formula that failed evaluation."""
    prompt = build_fix_formula_prompt(formula, error_msg)
    try:
        text = client.generate(prompt, temperature=temperature, max_tokens=4096)
    except Exception as e:
        print(f"    [FixFormula] LLM error: {e}")
        return None
    blocks = parse_formula_blocks(text)
    return blocks[0] if blocks else None


# ---------------------------------------------------------------------------
# Validate / fit (copied from mle_bo.py, split into two phases)
# ---------------------------------------------------------------------------

# Re-import the verify functions so they're available without touching v1
from .mle_bo import (
    _verify_kernel_agnostic_to_n,
    _verify_kernel_differentiable,
    _verify_kernel_psd,
    _model_selection_score,
    _crps_bic_score,
    compute_log_prior,
)


def validate_kernel_code(
    kernel_code_str: str,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    train_x_sample: torch.Tensor,
    bounds: list | None = None,
) -> tuple[bool, str, str]:
    """Fast validation only (Stage 1-2): parse, instantiate, agnostic/grad/psd checks.

    Returns (ok, fail_reason, tb_str).
    ok=True means code is safe to fit. ok=False means skip fitting.
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * input_dim

    # --- Stage 1: parse & instantiate ---
    try:
        kernel_instance = _kernel_from_code(
            kernel_code_str, input_dim, device, dtype, bounds=bounds, wrap_scale=False
        )
    except SyntaxError:
        return False, "syntax_error", _traceback.format_exc()
    except KeyError as e:
        msg = str(e)
        if "EvolvedKernel" in msg or "Kernel subclass" in msg:
            return False, "no_kernel_class", _traceback.format_exc()
        return False, "instantiation_error", _traceback.format_exc()
    except Exception:
        return False, "instantiation_error", _traceback.format_exc()

    if kernel_instance is None:
        return False, "instantiation_error", "kernel_from_code returned None"

    # --- Stage 2: sanity checks ---
    x_sample = train_x_sample[:10].to(device)
    agnostic_ok, agnostic_diag = _verify_kernel_agnostic_to_n(kernel_instance, x_sample, device)
    if not agnostic_ok:
        return False, "agnostic_check", agnostic_diag

    grad_ok, grad_diag = _verify_kernel_differentiable(kernel_instance, x_sample[:3], device)
    if not grad_ok:
        return False, "grad_check", grad_diag

    if not _verify_kernel_psd(kernel_instance, x_sample, device):
        return False, "psd_check", "Cholesky decomposition failed (not PSD)"

    return True, "", ""


def fit_and_score_kernel(
    kernel_code_str: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model_selection: str = "loss",
    state_dict: dict | None = None,
    skip_fit: bool = False,
    timeout: float = 120.0,
    bounds: list | None = None,
    score_subsample: float = 1.0,
    num_eval: int = 100,
) -> tuple:
    """Fit GP model and compute selection score (Stage 3-4).

    Call this only for kernels that passed validate_kernel_code.
    Returns (score, model, eval_time, fail_reason, tb_str, nll, timing).
    Same 7-tuple contract as mle_bo.evaluate_kernel_code.

    num_eval: if > 0, LOOCV / rank scoring uses at most this many training points
    with the largest y (model space: higher is better). If n <= num_eval, all points
    are used. If <= 0, no top-y restriction (use all points after score_subsample).

    Timeout is enforced via SIGALRM when available (Linux/macOS, main thread), which
    immediately interrupts C extensions such as scipy L-BFGS-B. Falls back to a
    ThreadPoolExecutor-based timeout otherwise (cannot interrupt C extensions).
    """
    import signal
    import threading
    import concurrent.futures

    start_eval = time.perf_counter()
    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
    device = train_x.device
    n = train_x.shape[0]
    input_dim = train_x.shape[1]
    if bounds is None:
        bounds = [(0.0, 1.0)] * input_dim

    def _fit_and_score():
        """Stage 3 (fit) + Stage 4 (scoring) — all inside timeout."""
        # --- Stage 3: model init & fit ---
        _t_fit = time.perf_counter()
        try:
            Y_train = train_y.unsqueeze(-1) if train_y.dim() == 1 else train_y
            model = _initialize_model(
                train_x, Y_train, base_kernel=kernel_code_str,
                device=device, dtype=train_x.dtype, bounds=bounds,
            )
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
            final_loss = _fit_model(model, return_loss=True, skip_fit=skip_fit)
        except Exception:
            t_fit_sec = time.perf_counter() - _t_fit
            return float("inf"), None, "fit_error", _traceback.format_exc(), t_fit_sec, float("inf")
        t_fit_sec = time.perf_counter() - _t_fit

        if final_loss is None or math.isnan(final_loss) or model is None:
            return float("inf"), None, "nan_loss", None, t_fit_sec, float("inf")

        # --- Stage 4: model selection scoring ---
        try:
            tx = model.train_inputs[0]
            ty = model.train_targets.squeeze(-1)

            # Subsample for LOOCV-based scoring (adds noise, reduces compute)
            if 0.0 < score_subsample < 1.0 and tx.shape[0] > 10:
                sub_n = max(int(tx.shape[0] * score_subsample), 10)
                if sub_n < tx.shape[0]:
                    perm = torch.randperm(tx.shape[0], device=tx.device)[:sub_n]
                    tx = tx[perm]
                    ty = ty[perm]

            # Score only on the highest-y points (BO targets large y in model space)
            if num_eval > 0 and tx.shape[0] > num_eval:
                top_idx = torch.topk(ty, k=num_eval, largest=True).indices
                tx = tx[top_idx]
                ty = ty[top_idx]

            if model_selection in ("nlpd", "nlpd_prior"):
                with torch.no_grad():
                    score = loocv_nlpd_loss(model, model.likelihood, tx, ty).item()
            elif model_selection in ("crps", "crps_prior"):
                with torch.no_grad():
                    score = loocv_crps_loss(model, model.likelihood, tx, ty).item()
            elif model_selection in ("rank_based", "rank_based_prior"):
                with torch.no_grad():
                    score = rank_weighted_loss(model, model.likelihood, tx, ty).item()
            elif model_selection == "crps_bic":
                with torch.no_grad():
                    crps = loocv_crps_loss(model, model.likelihood, tx, ty).item()
                n_loo = tx.shape[0]
                score = _crps_bic_score(crps, n_loo, model)
            else:
                score = _model_selection_score(final_loss, model, n, model_selection)

            if model_selection.endswith("_prior"):
                score = score - compute_log_prior(model, model.likelihood).item()
        except Exception:
            return float("inf"), None, "fit_error", _traceback.format_exc(), t_fit_sec, float("inf")

        return score, model, None, None, t_fit_sec, final_loss

    # ------------------------------------------------------------------
    # Timeout enforcement
    #
    # Prefer SIGALRM (Unix only, must be called from the main thread).
    # SIGALRM is a real OS signal that interrupts C extensions immediately
    # (e.g. scipy L-BFGS-B), unlike a ThreadPoolExecutor timeout which
    # only fires at the Python level and cannot kill a running thread.
    # ------------------------------------------------------------------
    _use_signal = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )

    if _use_signal:
        _alarm_secs = max(1, int(math.ceil(timeout)))

        class _AlarmTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _AlarmTimeout(f"exceeded timeout of {timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(_alarm_secs)
        try:
            score, model, fail_reason, tb_str, t_fit_sec, nll = _fit_and_score()
        except _AlarmTimeout:
            return float("inf"), None, 0.0, "timeout", f"exceeded timeout of {timeout}s", float("inf"), {}
        except Exception:
            return float("inf"), None, 0.0, "outer_error", _traceback.format_exc(), float("inf"), {}
        finally:
            # Always disarm the alarm and restore the old handler, even on
            # return from an except block (finally runs before the return).
            signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass
    else:
        # Fallback: ThreadPoolExecutor.  Note that this cannot forcibly kill
        # threads running C extensions; it only fires at the Python level.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_fit_and_score)
        try:
            score, model, fail_reason, tb_str, t_fit_sec, nll = future.result(timeout=timeout)
            executor.shutdown(wait=True)
        except (concurrent.futures.TimeoutError, TimeoutError):
            executor.shutdown(wait=False, cancel_futures=True)
            return float("inf"), None, 0.0, "timeout", f"exceeded timeout of {timeout}s", float("inf"), {}
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            return float("inf"), None, 0.0, "outer_error", _traceback.format_exc(), float("inf"), {}

    if fail_reason is not None:
        return float("inf"), None, 0.0, fail_reason, tb_str, nll, {}

    eval_time = time.perf_counter() - start_eval
    print(f"    [Evaluation] score={score:.4f}, nll={nll:.4f}, eval_time={eval_time:.4f}s")
    timing = {"fit_sec": round(t_fit_sec, 4)}
    return score, model, eval_time, None, None, nll, timing
