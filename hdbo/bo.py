"""
Bayesian optimization for high-dimensional benchmarks.
Mimics reference: SingleTaskGP, fit_gpytorch_mll_scipy, standardize(Y), gen_candidates_scipy.
Self-contained: no imports from blackbox_opt.
"""
import gc
import math
import os
import time

# Set memory allocator configuration to prevent fragmentation before torch is imported
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from contextlib import ExitStack

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior

try:
    import botorch
    from botorch.acquisition import (
        qExpectedImprovement,
        qLogExpectedImprovement,
        qLogNoisyExpectedImprovement,
        qUpperConfidenceBound
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling, gen_candidates_scipy, gen_candidates_torch
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.utils import standardize
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False


# Backend switches (scipy | torch). Temporary knobs for fit/acqf optimizer ablation.
# Flip via set_backends() from a runner; defaults preserve original behavior.
_FIT_BACKEND = "scipy"
_ACQF_BACKEND = "scipy"


def set_backends(fit: str = None, acqf: str = None):
    global _FIT_BACKEND, _ACQF_BACKEND
    if fit is not None:
        assert fit in ("scipy", "torch"), fit
        _FIT_BACKEND = fit
    if acqf is not None:
        assert acqf in ("scipy", "torch"), acqf
        _ACQF_BACKEND = acqf


def _clear_gpu_memory():
    """Release GPU memory aggressively by triggering synchronization, GC, and cache clearing."""
    for _ in range(2):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    # Final 'hard' clear
    if torch.cuda.is_available():
        with torch.cuda.device(torch.cuda.current_device()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # Clear any shared memory remnants


def _bounds_tensor(bounds, device=None, dtype=torch.float64):
    """Convert list of (lo, hi) to (2, d) tensor."""
    lo = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    hi = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)
    return torch.stack([lo, hi])

LINEAR_KERNEL_CODE = """
import torch
import gpytorch
from torch.nn import Parameter

class EvolvedKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def __init__(self, ard_num_dims: int, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name="raw_variance", parameter=Parameter(torch.tensor(0.0)))
    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        variance = self.raw_variance
        covar = variance * torch.matmul(x1, x2.transpose(-1, -2))
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

PERIODIC_KERNEL_CODE = """
import torch
import gpytorch
import math
from torch.nn import Parameter
from gpytorch.constraints import Positive

class EvolvedKernel(gpytorch.kernels.Kernel):
    is_stationary = True
    def __init__(self, ard_num_dims: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = ard_num_dims
        init = torch.zeros(ard_num_dims)
        self.register_parameter(name="raw_lengthscale", parameter=Parameter(init))
        self.register_parameter(name="raw_period", parameter=Parameter(torch.zeros(ard_num_dims)))

        self.register_constraint("raw_lengthscale", Positive())
        self.register_constraint("raw_period", Positive())
    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        lengthscale = self.raw_lengthscale
        period = self.raw_period
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        scaled_diff = torch.sin(math.pi * diff / period) / lengthscale
        dist_sq = (scaled_diff ** 2).sum(-1)
        covar = torch.exp(-2.0 * dist_sq)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

DSP_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
    
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist_sq = torch.cdist(x1s, x2s, p=2).pow(2)
        covar = torch.exp(-0.5 * dist_sq)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

MATERN52_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
    
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist = torch.cdist(x1s, x2s, p=2).clamp(min=1e-15)
        sqrt5_d = math.sqrt(5.0) * dist
        covar = (1.0 + sqrt5_d + (5.0 / 3.0) * dist.pow(2)) * torch.exp(-sqrt5_d)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

MATERN32_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist = torch.cdist(x1s, x2s, p=2).clamp(min=1e-15)
        sqrt3_d = math.sqrt(3.0) * dist
        covar = (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

RQ_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):

        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )

        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )
        
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        alpha_constraint = Positive()
        self.register_constraint("raw_alpha", alpha_constraint)


    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        alpha = self.alpha
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist_sq = torch.cdist(x1s, x2s, p=2).pow(2)
        covar = (1.0 + dist_sq / (2.0 * alpha)) ** (-alpha)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

DSP_SINGLE_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
    
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )
        super().__init__(
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist_sq = torch.cdist(x1s, x2s, p=2).pow(2)
        covar = torch.exp(-0.5 * dist_sq)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

MATERN52_SINGLE_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
    
        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )
        super().__init__(
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist = torch.cdist(x1s, x2s, p=2).clamp(min=1e-15)
        sqrt5_d = math.sqrt(5.0) * dist
        covar = (1.0 + sqrt5_d + (5.0 / 3.0) * dist.pow(2)) * torch.exp(-sqrt5_d)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

RQ_SINGLE_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
    def __init__(self, *,
        ard_num_dims: int,
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):

        lengthscale_prior = LogNormalPrior(
            loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
        )
        lengthscale_constraint = GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        )

        super().__init__(
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )
        
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        alpha_constraint = Positive()
        self.register_constraint("raw_alpha", alpha_constraint)


    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)
        alpha = self.alpha
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale
        dist_sq = torch.cdist(x1s, x2s, p=2).pow(2)
        covar = (1.0 + dist_sq / (2.0 * alpha)) ** (-alpha)
        if diag: return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""



SPHERICAL_KERNEL_CODE = """
import math
from typing import Sequence

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(
        self,
        *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
        if ard_num_dims == 1:
            raise ValueError(
                "ard_num_dims must be equal to the dimensionality of the input data."
            )
        if isinstance(bounds[0], (int, float)):
            bounds = [(bounds[0], bounds[1])] * ard_num_dims

        if prior == "dsp_unscaled":
            loc = math.sqrt(2.0) + math.log(1) * 0.5
            scale = math.sqrt(3.0)
            lengthscale_prior = LogNormalPrior(loc=loc, scale=scale)
            
            # Mode calculation for LogNormal: exp(loc - scale^2)
            mode = math.exp(loc - scale**2)
            
            # Interval(min, inf) is mathematically identical to GreaterThan(min)
            lengthscale_constraint = GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mode
            )
            
        elif prior == "gamma_3_6":
            alpha, beta = 3.0, 6.0
            lengthscale_prior = GammaPrior(alpha, beta)
            
            # Mode calculation for Gamma: (alpha - 1) / beta
            mode = (alpha - 1.0) / beta
            
            lengthscale_constraint = GreaterThan(
                1e-2, transform=None, initial_value=lengthscale_prior.mode
            )
        else:
            raise ValueError("Unknown prior. Use 'dsp_unscaled' or 'gamma_3_6'.")

        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

        _dtype = self.raw_lengthscale.dtype
        _bounds = torch.tensor(bounds, dtype=_dtype)
        self.register_buffer("_mins", _bounds[..., 0])
        self.register_buffer("_maxs", _bounds[..., 1])
        self.register_buffer("_centers", (self._mins + self._maxs).div(2.0))
        assert torch.all(self._maxs > self._mins), 'Invalid bounds'

        coeffs = torch.zeros(2, dtype=_dtype)
        self.register_parameter("raw_coeffs", torch.nn.Parameter(coeffs))

        glob_ls = torch.zeros(1, dtype=_dtype)
        self.register_parameter("raw_glob_ls", torch.nn.Parameter(glob_ls))

    def project_onto_unit_sphere(self, x):
        x_sq_norm = x.square().sum(dim=-1, keepdim=True)
        x_ = torch.cat([2 * x, (x_sq_norm - 1.0)], dim=-1).mul(
            1.0 / (1.0 + x_sq_norm)
        )
        return x_

    @property
    def coeffs(self):
        return torch.nn.functional.softmax(self.raw_coeffs, dim=-1)

    @property
    def glob_ls(self):
        return torch.sigmoid(self.raw_glob_ls)

    def forward(self, x1, x2, diag=False, **params):
        x1_equal_x2 = torch.equal(x1, x2)

        assert torch.all(x1 <= self._maxs)
        assert torch.all(x1 >= self._mins)
        assert torch.all(x2 <= self._maxs)
        assert torch.all(x2 >= self._mins)

        lengthscale = self.lengthscale
        max_sq_norm = (
            (self._maxs - self._mins)[..., None, :]
            .div(2.0 * lengthscale)
            .square()
            .sum(dim=-1, keepdim=True)
        )
        glob_ls = torch.sqrt(self.glob_ls * max_sq_norm)

        x1 = x1.sub(self._centers).div(lengthscale)
        x2 = x1 if x1_equal_x2 else x2.sub(self._centers).div(lengthscale)

        x1 = x1.div(glob_ls)
        x2 = x2.div(glob_ls)

        x1_ = self.project_onto_unit_sphere(x1)
        x2_ = self.project_onto_unit_sphere(x2)

        terms = self.coeffs
        term0_sqrt = terms[0].sqrt()
        term1_sqrt = terms[1].sqrt()
        x1_ = torch.cat([x1_ * term1_sqrt, term0_sqrt.expand_as(x1_[..., :1])], dim=-1)

        if x1_equal_x2:
            kernel = x1_ @ x1_.mT
        else:
            x2_ = torch.cat([x2_ * term1_sqrt, term0_sqrt.expand_as(x2_[..., :1])], dim=-1)
            kernel = x1_ @ x2_.mT

        if diag:
            return kernel.diagonal(dim1=-2, dim2=-1)
            
        return kernel
"""

CYLINDRICAL_KERNEL_CODE = """
import math
from typing import Sequence

import torch
from torch import Tensor
import gpytorch
from gpytorch.constraints import GreaterThan, Positive, Interval
from gpytorch.priors import Prior
from gpytorch import settings

class EvolvedKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        *,
        ard_num_dims: int,
        bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
        num_angular_weights: int = 4,
        eps: float = 1e-6,
        angular_weights_prior: Prior | None = None,
        angular_weights_constraint: Interval | None = None,
        alpha_prior: Prior | None = None,
        alpha_constraint: Interval | None = None,
        beta_prior: Prior | None = None,
        beta_constraint: Interval | None = None,
        **kwargs,
    ):
        
        if angular_weights_constraint is None:
            angular_weights_constraint = Positive()

        if alpha_constraint is None:
            alpha_constraint = Positive()

        if beta_constraint is None:
            beta_constraint = Positive()

        super().__init__(**kwargs)

        self.num_angular_weights = num_angular_weights
        self.radial_base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        self.eps = eps
        
        if isinstance(bounds[0], (int, float)):
            bounds = [(bounds[0], bounds[1])] * ard_num_dims
        _bounds = torch.tensor(bounds)
        mins = _bounds[..., 0]
        maxs = _bounds[..., 1]
        centers = (mins + maxs).div(2.0)
        half_ranges = (maxs - mins).div(2.0)
        radius = half_ranges.square().sum().clamp(min=1e-30).sqrt()
        self.register_buffer("_center", centers)
        self.register_buffer("_radius", radius)

        self.register_parameter(
            name="raw_angular_weights",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, num_angular_weights)),
        )
        self.register_constraint("raw_angular_weights", angular_weights_constraint)
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_beta", beta_constraint)

        if angular_weights_prior is not None:
            if not isinstance(angular_weights_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(angular_weights_prior).__name__)
            self.register_prior(
                "angular_weights_prior",
                angular_weights_prior,
                lambda m: m.angular_weights,
                lambda m, v: m._set_angular_weights(v),
            )
        if alpha_prior is not None:
            if not isinstance(alpha_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(alpha_prior).__name__)
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha, lambda m, v: m._set_alpha(v))
        if beta_prior is not None:
            if not isinstance(beta_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(beta_prior).__name__)
            self.register_prior("beta_prior", beta_prior, lambda m: m.beta, lambda m, v: m._set_beta(v))

    @property
    def angular_weights(self) -> Tensor:
        return self.raw_angular_weights_constraint.transform(self.raw_angular_weights)

    @angular_weights.setter
    def angular_weights(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.initialize(raw_angular_weights=self.raw_angular_weights_constraint.inverse_transform(value))

    @property
    def alpha(self) -> Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: Tensor) -> None:
        self._set_alpha(value)

    def _set_alpha(self, value: Tensor | float) -> None:
        # Used by the alpha_prior
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self) -> Tensor:
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value: Tensor) -> None:
        self._set_beta(value)

    def _set_beta(self, value: Tensor | float) -> None:
        # Used by the beta_prior
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool | None = False, **params) -> Tensor:
        center = self._center.to(device=x1.device, dtype=x1.dtype)
        radius = self._radius.to(device=x1.device, dtype=x1.dtype).clamp_min(self.eps)
        x1_, x2_ = x1.sub(center).div(radius), x2.sub(center).div(radius)

        x1_, x2_ = x1_.clone(), x2_.clone()
        # Jitter datapoints that are exactly 0
        x1_[x1_ == 0], x2_[x2_ == 0] = x1_[x1_ == 0] + self.eps, x2_[x2_ == 0] + self.eps
        r1 = x1_.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        r2 = x2_.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        # allow small numerical overshoots after scaling
        if torch.any(r1 > (1.0 + 1e-6)) or torch.any(r2 > (1.0 + 1e-6)):
            raise RuntimeError("Cylindrical kernel not defined for data points with radius > 1. Scale your data!")

        a1, a2 = x1_.div(r1), x2_.div(r2)
        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1)).clamp(min=-1.0, max=1.0)
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1).clamp(min=-1.0, max=1.0)
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None].mul(gram_mat.pow(p))

        with settings.lazily_evaluate_kernels(False):
            radial_kernel = self.radial_base_kernel(self.kuma(r1), self.kuma(r2), diag=diag, **params)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x: Tensor) -> Tensor:
        alpha = self.alpha.view(*self.batch_shape, 1, 1)
        beta = self.beta.view(*self.batch_shape, 1, 1)

        # x should be a radius in [0, 1]. Clamp for numeric safety.
        x = x.clamp(min=0.0, max=1.0)
        t = (1.0 - x.pow(alpha)).clamp(min=0.0).add(self.eps)
        res = 1.0 - t.pow(beta)
        return res.clamp(min=0.0, max=1.0)

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        return self.radial_base_kernel.num_outputs_per_input(x1, x2)
"""


DISCOVERY_1_KERNEL_CODE = """
import math
import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.priors.torch_priors import LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    # Notice has_lengthscale=False here because we are explicitly implementing a decoupled ARD strategy
    def __init__(self, ard_num_dims: int, poly_order: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.poly_order = poly_order
        
        # 1. Global lengthscale (magnitude)
        self.register_parameter("raw_global_lengthscale", torch.nn.Parameter(torch.tensor(0.0)))
        self.register_constraint("raw_global_lengthscale", GreaterThan(1e-4))
        
        # 2. Decoupled ARD sensitivities (direction)
        self.register_parameter("raw_ard_weights", torch.nn.Parameter(torch.zeros(ard_num_dims)))
        
        # 3. Polynomial coefficients
        self.register_parameter("raw_coeffs", torch.nn.Parameter(torch.zeros(poly_order + 1)))

    @property
    def global_lengthscale(self):
        return self.raw_global_lengthscale_constraint.transform(self.raw_global_lengthscale)

    def _project_to_sphere(self, x, D):
        # Apply decoupled ARD weights and global lengthscale scaling O(sqrt(D))
        weights = torch.exp(self.raw_ard_weights) # Ensure positive
        z = (x * weights / self.global_lengthscale) * math.sqrt(3.0 / D)
        
        z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)
        denominator = (z_norm_sq + 1.0).clamp(min=1e-15)
        # Inverse stereographic projection
        return torch.cat([2 * z, z_norm_sq - 1.0], dim=-1) / denominator

    def forward(self, x1, x2, diag=False, **params):
        D = x1.shape[-1]
        
        P1 = self._project_to_sphere(x1, D)
        P2 = self._project_to_sphere(x2, D)
        
        # Dot product on the hypersphere
        dot = torch.matmul(P1, P2.transpose(-1, -2))
        
        # Simplex constraint to remove implicit outputscale
        coeffs = torch.nn.functional.softmax(self.raw_coeffs, dim=0)
        
        covar = torch.zeros_like(dot)
        for i in range(self.poly_order + 1):
            covar = covar + coeffs[i] * (dot ** i)
            
        if diag: 
            return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

DISCOVERY_2_KERNEL_CODE = """
import math
import torch
import gpytorch
from torch import Tensor
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors import LogNormalPrior

class EvolvedKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, ard_num_dims: int, **kwargs):
        lengthscale_prior = LogNormalPrior(loc=math.sqrt(2.0), scale=math.sqrt(3.0))
        lengthscale_constraint = GreaterThan(2.5e-2)

        super().__init__(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            **kwargs
        )

        # RBF lengthscale modifier (Note: mathematically redundant with ARD, but kept for structural fidelity)
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.tensor(0.5)))
        self.register_constraint("raw_alpha", Positive())

        # Parameter controlling the "squishiness" of the boundary space
        self.register_parameter(name="raw_warp_param", parameter=torch.nn.Parameter(torch.tensor(0.8)))
        self.register_constraint("raw_warp_param", Positive())

        self.register_parameter(name="raw_poly_scale", parameter=torch.nn.Parameter(torch.tensor(1.2)))
        self.register_constraint("raw_poly_scale", Positive())

    @property
    def alpha(self) -> Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @property
    def warp_param(self) -> Tensor:
        return self.raw_warp_param_constraint.transform(self.raw_warp_param)

    @property
    def poly_scale(self) -> Tensor:
        return self.raw_poly_scale_constraint.transform(self.raw_poly_scale)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if x1.dim() == 1: x1 = x1.unsqueeze(-1)
        if x2.dim() == 1: x2 = x2.unsqueeze(-1)

        # 1. Base ARD Scaling
        ls = self.lengthscale.view(1, -1)
        x1_, x2_ = x1.div(ls), x2.div(ls)
        
        # 2. Compute Base Geometries (for standard terms)
        dist_sq = torch.cdist(x1_, x2_, p=2).pow(2)
        dot_product = torch.matmul(x1_, x2_.transpose(-1, -2))

        # 3. Standard Stationary & Polynomial Terms
        rbf_term = torch.exp(-0.5 * dist_sq / self.alpha**2)
        poly_term = (self.poly_scale * dot_product + 1).pow(3)
        
        # Safe Matern computation to avoid NaN gradients
        sqrt5_d = math.sqrt(5.0) * dist_sq.clamp(min=1e-15).sqrt()
        matern_term = (1.0 + sqrt5_d + (5.0 / 3.0) * dist_sq) * torch.exp(-sqrt5_d)
        
        # 4. THE RIGOROUS FIX: Input Warped Non-Stationary Term
        # We squash the ARD-scaled inputs using a hyperbolic tangent.
        # This compresses the boundaries of the high-dimensional space.
        w_x1 = torch.tanh(self.warp_param * x1_)
        w_x2 = torch.tanh(self.warp_param * x2_)
        
        w_dist_sq = torch.cdist(w_x1, w_x2, p=2).pow(2)
        
        # Apply a standard RBF on the warped space
        warped_term = torch.exp(-0.5 * w_dist_sq)

        # 5. Composite Covariance
        covar = rbf_term + poly_term * matern_term + warped_term

        if diag:
            return covar.diagonal(dim1=-1, dim2=-2)
        return covar
"""

def _kernel_from_code(kernel_code_str, input_dim, device, dtype=torch.float64, bounds=None, wrap_scale=True):
    """Build gpytorch kernel from code string defining EvolvedKernel class."""
    import math
    from torch.nn import Parameter
    from torch.nn.functional import softplus, softmax
    from typing import Sequence, List, Tuple, Optional, Union
    exec_namespace = {
        "torch": torch,
        "gpytorch": gpytorch,
        "math": math,
        "Parameter": Parameter,
        "softplus": softplus,
        "softmax": softmax,
        # typing — LLM often uses these in type hints without importing
        "Sequence": Sequence,
        "List": List,
        "Tuple": Tuple,
        "Optional": Optional,
        "Union": Union,
    }
    # Prepend typing imports so they're available as real module-level imports
    # (pre-seeding exec_namespace alone can fail for class-body annotation evaluation)
    typing_header = "from typing import Sequence, List, Tuple, Optional, Union\n"
    exec(typing_header + kernel_code_str, exec_namespace)
    if "EvolvedKernel" in exec_namespace:
        EvolvedKernelClass = exec_namespace["EvolvedKernel"]
    else:
        # Fallback: find any gpytorch.kernels.Kernel subclass defined in the code
        kernel_classes = [
            v for k, v in exec_namespace.items()
            if isinstance(v, type) and issubclass(v, gpytorch.kernels.Kernel) and v is not gpytorch.kernels.Kernel
        ]
        if not kernel_classes:
            raise KeyError("No EvolvedKernel or Kernel subclass found in code")
        EvolvedKernelClass = kernel_classes[0]
    bounds = bounds if bounds is not None else [(0.0, 1.0)] * input_dim
    # ard_num_dims = None if input_dim > 500 else input_dim
    ard_num_dims = input_dim
    try:
        k = EvolvedKernelClass(ard_num_dims=ard_num_dims, bounds=bounds)
    except TypeError:
        try:
            k = EvolvedKernelClass(ard_num_dims=ard_num_dims)
        except TypeError:
            try:
                k = EvolvedKernelClass(input_dim=input_dim)
            except TypeError:
                k = EvolvedKernelClass()
    k = k.to(device=device, dtype=dtype)
    # Move plain tensor attributes that .to() misses (not registered as buffer/parameter)
    for attr, val in vars(k).items():
        if isinstance(val, torch.Tensor):
            setattr(k, attr, val.to(device=device, dtype=dtype))
    if wrap_scale:
        k = gpytorch.kernels.ScaleKernel(k)
    return k
    # except Exception:
    #     raise ValueError("Failed to build kernel from code. Code must define class EvolvedKernel.")


def _make_kernel(base_kernel, input_dim, device, dtype=torch.float64, bounds=None):
    """Build gpytorch kernel from name or code string."""
    # Custom kernel from file
    if isinstance(base_kernel, str) and base_kernel.startswith("file:"):
        path = base_kernel[5:].strip()
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        return _kernel_from_code(code, input_dim, device, dtype, bounds=bounds)
    # Inline kernel code (defines EvolvedKernel)
    if isinstance(base_kernel, str) and "EvolvedKernel" in base_kernel and "\n" in base_kernel:
        return _kernel_from_code(base_kernel, input_dim, device, dtype, bounds=bounds)

    base = (base_kernel or "dsp").lower()
    if base == "matern52":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(MATERN52_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "matern32":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(MATERN32_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "rq":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(RQ_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "dsp":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(DSP_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "spherical":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(SPHERICAL_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "cylindrical":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(CYLINDRICAL_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "linear":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(LINEAR_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "periodic":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(PERIODIC_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "discovery_1":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(DISCOVERY_1_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "discovery_2":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(DISCOVERY_2_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base in ["dsp_single", "rbf_single"]:
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(DSP_SINGLE_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base in ["matern52_single", "matern_single"]:
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(MATERN52_SINGLE_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    elif base == "rq_single":
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(RQ_SINGLE_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    else:
        # Fallback to DSP if an unsupported base name is given
        if bounds is None:
            bounds = [(0.0, 1.0)] * input_dim
        k = _kernel_from_code(DSP_KERNEL_CODE, input_dim, device, dtype, bounds=bounds, wrap_scale=False)
    return gpytorch.kernels.ScaleKernel(k).to(device=device, dtype=dtype)


def _initialize_model(X_train, Y_train, base_kernel, device, dtype=torch.float64, bounds=None):
    """Initialize SingleTaskGP with likelihood prior/constraint (mimics reference)."""
    d = X_train.size(-1)
    mean = gpytorch.means.ConstantMean()
    covar = _make_kernel(base_kernel, d, device, dtype, bounds=bounds)
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    )
    model = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        mean_module=mean,
        covar_module=covar,
        likelihood=likelihood,
    ).to(device=device, dtype=dtype)

    # GPyTorch constraint lower/upper bounds are plain tensors (not registered buffers),
    # so .to(device) above doesn't move them. Fix explicitly to avoid device mismatch
    # in sample_all_priors when botorch retries fitting after a NotPSDError.
    for module in model.modules():
        for constraint in getattr(module, "_constraints", {}).values():
            for attr in ("_lower_bound", "_upper_bound"):
                val = getattr(constraint, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(constraint, attr, val.to(device=device, dtype=dtype))

    return model


def _fit_model(model, return_loss=False, skip_fit=False):
    """Fit GP via fit_gpytorch_mll_scipy with gpytorch settings (mimics reference).
    If return_loss=True, returns the final NLL (scalar float) using model's train data."""
    model.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)

    if not skip_fit:
        import botorch.optim.utils.model_utils as _mu
        import botorch.fit as _bf
        _orig_sample_all_priors = _mu.sample_all_priors

        _model_device = next(model.parameters()).device

        def _patched_sample_all_priors(model_, **kwargs):
            # prior.sample() always returns CPU tensors even after prior.to(device).
            # Patch setting_closure calls to move sampled values to the model's device.
            if _model_device.type == "cpu":
                return _orig_sample_all_priors(model_, **kwargs)
            for _name, (_prior, _closure, _setting_closure) in getattr(model_, "_priors", {}).items():
                if _setting_closure is None:
                    continue
                try:
                    _sampled = _prior.sample(_closure(model_).shape).to(_model_device)
                    _setting_closure(model_, _sampled)
                except Exception:
                    pass
            for _child in model_.children():
                _patched_sample_all_priors(_child)

        # botorch.fit._fit_fallback imports sample_all_priors directly, so patch both
        _mu.sample_all_priors = _patched_sample_all_priors
        _bf.sample_all_priors = _patched_sample_all_priors
        try:
            with ExitStack() as es:
                es.enter_context(gpytorch.settings.cholesky_max_tries(10))
                es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
                es.enter_context(
                    gpytorch.settings.fast_computations(log_prob=True, covar_root_decomposition=False, solves=False)
                )
                _fit_optimizer = (
                    botorch.fit.fit_gpytorch_mll_torch
                    if _FIT_BACKEND == "torch"
                    else botorch.fit.fit_gpytorch_mll_scipy
                )
                fit_gpytorch_mll(mll, optimizer=_fit_optimizer)
        finally:
            _mu.sample_all_priors = _orig_sample_all_priors
            _bf.sample_all_priors = _orig_sample_all_priors

    model.eval()
    if return_loss:
        with torch.no_grad():
            output = model(model.train_inputs[0])
            return (-mll(output, model.train_targets)).mean().item()
    return None


def _optimize_acquisition_with_fallback(
    model,
    bounds,
    acquisition,
    f_best,
    device,
    dtype,
    batch_size=1,
    X_baseline=None,
    verbose=True,
    prune_baseline=False,
    seed=None,
    sequential=False,
    logger=None,
    bo_iter=None,
    n_evals=None,
    best_kernel_source=None,
    raw_samples=512,
    num_restarts=4,
    mc_samples=256,
    maxiter=300,
    batch_limit=64,
):
    """
    Optimize acquisition with fallback: if optimize_acqf fails,
    return random points generated with Sobol sequences.
    Returns (x_next, failed_flag).
    """
    import traceback as _tb_mod
    try:
        return (
            _optimize_acquisition_botorch(
                model, bounds, acquisition, f_best, device, dtype,
                batch_size=batch_size, X_baseline=X_baseline,
                prune_baseline=prune_baseline, sequential=sequential,
                raw_samples=raw_samples, num_restarts=num_restarts,
                mc_samples=mc_samples, maxiter=maxiter, batch_limit=batch_limit,
            ),
            False,
        )
    except Exception as e:
        tb_str = _tb_mod.format_exc()
        error_type = type(e).__name__
        error_msg = str(e)
        if verbose:
            print(f"  optimize_acqf failed [{error_type}: {error_msg}]. Generating random points using Sobol sequences.")
            print(tb_str)
        if logger is not None:
            logger.log_acqf_error(
                bo_iter=bo_iter,
                n_evals=n_evals,
                acquisition=acquisition,
                error_type=error_type,
                error_msg=error_msg,
                tb=tb_str,
                best_kernel_source=best_kernel_source,
            )
        dim = len(bounds)
        # Use n_evals in seed so each fallback generates different points
        _offset = (n_evals or 0) + 99999
        fallback_seed = seed + _offset if seed is not None else _offset
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=fallback_seed)
        x_next = sobol.draw(batch_size).to(device=device, dtype=dtype)
        for d in range(dim):
            lo, hi = bounds[d]
            x_next[:, d] = x_next[:, d] * (hi - lo) + lo
        return x_next, True


def _optimize_acquisition_botorch(model, bounds, acquisition, f_best, device, dtype, batch_size=1, X_baseline=None, prune_baseline=False, sequential=False,
                                   raw_samples=512, num_restarts=4, mc_samples=256, maxiter=300, batch_limit=64):
    """Use BoTorch to optimize acquisition. Returns (batch_size, d) tensor."""
    # Move model to CPU to allow full defragmentation of the GPU memory
    orig_device = next(model.parameters()).device
    model.to(device=torch.device("cpu"))
    _clear_gpu_memory()
    model.to(device=orig_device)

    # Freeze model parameters to save memory during acquisition optimization
    for param in model.parameters():
        param.requires_grad = False

    bounds_t = _bounds_tensor(bounds, device=device, dtype=dtype)
    d = bounds_t.shape[-1]

    acq_options = {
        "raw_samples": raw_samples,
        "num_restarts": num_restarts,
        "retry_on_optimization_warning": False,
        "options": {
            "nonnegative": False,
            "sample_around_best": True,
            "sample_around_best_sigma": 0.1,
            "maxiter": maxiter,
            "batch_limit": batch_limit,
        },
    }

    if acquisition == "ts":
        sobol = torch.quasirandom.SobolEngine(dimension=bounds_t.shape[-1], scramble=True)
        X_pool = sobol.draw(512).to(device=device, dtype=dtype)
        lo, hi = bounds_t[0], bounds_t[1]
        X_pool = X_pool * (hi - lo) + lo
        X_pool = X_pool.unsqueeze(0)
        mps = MaxPosteriorSampling(model=model)
        candidates = mps(X_pool, num_samples=batch_size)
        return candidates.squeeze(0)

    acq_fn = None
    sampler = None
    if acquisition == "ei":
        best_f = torch.tensor(f_best, device=device, dtype=dtype)
        acq_fn = qExpectedImprovement(model=model, best_f=best_f)
    elif acquisition == "logei":
        best_f = torch.tensor(f_best, device=device, dtype=dtype)
        acq_fn = qLogExpectedImprovement(model=model, best_f=best_f)
    elif acquisition == "qlognei":
        if X_baseline is None:
            raise ValueError("qlognei requires X_baseline (observed points)")
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acq_fn = qLogNoisyExpectedImprovement(model=model, X_baseline=X_baseline, sampler=sampler, prune_baseline=prune_baseline)
    elif acquisition == "ucb":
        acq_fn = qUpperConfidenceBound(model=model, beta=2.0)
    elif acquisition == "pi":
        from botorch.acquisition.monte_carlo import qProbabilityOfImprovement
        best_f = torch.tensor(f_best, device=device, dtype=dtype)
        acq_fn = qProbabilityOfImprovement(model=model, best_f=best_f)
    elif acquisition == "logpi":
        from botorch.acquisition.analytic import LogProbabilityOfImprovement
        best_f = torch.tensor(f_best, device=device, dtype=dtype)
        acq_fn = LogProbabilityOfImprovement(model=model, best_f=best_f, maximize=True)
    elif acquisition == "posmean":
        from botorch.acquisition.analytic import PosteriorMean
        acq_fn = PosteriorMean(model=model, maximize=True)
    elif acquisition == "posstd":
        from botorch.acquisition.analytic import PosteriorStandardDeviation
        acq_fn = PosteriorStandardDeviation(model=model, maximize=True)
    elif acquisition == "qkg":
        from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
        acq_fn = qKnowledgeGradient(model=model, num_fantasies=4)
    elif acquisition == "qpes":
        from botorch.acquisition.utils import get_optimal_samples
        from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
        optimal_inputs, _ = get_optimal_samples(model=model.cpu(), bounds=bounds_t.cpu(), num_optima=4)
        acq_fn = qPredictiveEntropySearch(model=model.to(device), maximize=True, optimal_inputs=optimal_inputs.to(device))
    elif acquisition == "qmes":
        from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
        cand_set = bounds_t[0] + (bounds_t[1] - bounds_t[0]) * torch.rand(100, bounds_t.size(1), dtype=dtype, device=device)
        acq_fn = qLowerBoundMaxValueEntropy(model=model, candidate_set=cand_set, maximize=True)
    elif acquisition == "qjes":
        from botorch.acquisition.utils import get_optimal_samples
        from botorch.acquisition.joint_entropy_search import qJointEntropySearch
        optimal_inputs, optimal_outputs = get_optimal_samples(model=model.cpu(), bounds=bounds_t.cpu(), num_optima=4)
        acq_fn = qJointEntropySearch(model=model.to(device), optimal_inputs=optimal_inputs.to(device).to(dtype), optimal_outputs=optimal_outputs.to(device).to(dtype), estimation_type="LB")
    else:
        raise ValueError(f"Unsupported acquisition '{acquisition}'.")

    try:
        candidates, _ = optimize_acqf(
            acq_fn,
            bounds=bounds_t,
            q=batch_size,
            sequential=sequential,
            gen_candidates=gen_candidates_torch if _ACQF_BACKEND == "torch" else gen_candidates_scipy,
            **acq_options,
        )
        # gen_candidates_torch leaves requires_grad=True; scipy path already detaches.
        # Detach unconditionally so downstream f(x) callbacks with .numpy() work.
        return candidates.detach()
    finally:
        # Release acq_fn (holds model reference + X_baseline root decomposition cache)
        # and clear reserved CUDA pool even on exception paths to prevent gradual memory
        # growth when a kernel OOMs mid-optimization.
        del acq_fn
        del sampler
        _clear_gpu_memory()


def run_bo(
    benchmark_name: str,
    benchmarks: dict,
    dim: int = None,
    n_init: int = 5,
    n_iter: int = 20,
    batch_size: int = 1,
    n_candidates: int = 500,
    acquisition: str = "ei",
    base_kernel: str = "dsp",
    seed: int = 0,
    verbose: bool = True,
    save_dir: str = None,
    prune_baseline: bool = False,
    sequential: bool = False,
    on_eval=None,
):
    """
    Run BO on an HDBO benchmark. Returns (X_all, y_all, history).
    n_iter = total evaluations; num_rounds = ceil((n_iter - n_init) / batch_size).
    Benchmarks return value to MAXIMIZE (higher is better).
    """
    if not BOTORCH_AVAILABLE:
        raise RuntimeError("BoTorch is required. Install with: pip install botorch")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    info = benchmarks[benchmark_name]
    f = info["f"]
    # Ensure benchmark seed is set for reproduction (especially for MuJoCo)
    if hasattr(f, 'set_seed'):
        f.set_seed(seed)
    
    bounds = info["bounds"]
    dim = dim if dim is not None else info["dim"]
    if dim != info["dim"]:
        bounds_per_dim = info.get("bounds_per_dim")
        if bounds_per_dim is None:
            raise ValueError(f"Benchmark {benchmark_name} has fixed dimension {info['dim']}; --dim override not supported")
        bounds = [bounds_per_dim] * dim

    optimal_val = info.get("min_val")
    maximize = info.get("maximize", True)
    n_init = min(n_init, n_iter)

    # Initial points
    sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    X = sobol.draw(n_init).to(device=device, dtype=dtype)
    for d in range(dim):
        lo, hi = bounds[d]
        X[:, d] = X[:, d] * (hi - lo) + lo
    y = f(X).reshape(-1, 1).to(device=device, dtype=dtype)

    def best_val(y_slice):
        return y_slice.max().item() if maximize else y_slice.min().item()

    history = []
    for i in range(n_init):
        n_evals = i + 1
        best_so_far = best_val(y[:n_evals])
        history.append({"n_evals": n_evals, "best": best_so_far, "incumbent": best_so_far})
        if on_eval:
            regret = (best_so_far - optimal_val) if optimal_val is not None else None
            on_eval(n_evals, best_so_far, regret)
    if verbose:
        print(f"  Init: n_evals={n_init}, best={best_val(y):.4f}")

    n_remaining = n_iter - n_init
    num_rounds = max(0, (n_remaining + batch_size - 1) // batch_size) if n_remaining > 0 else 0

    history_path = os.path.join(save_dir, "history.txt") if save_dir else None
    if history_path:
        kernel_label = (
            os.path.splitext(os.path.basename(base_kernel[5:]))[0]
            if isinstance(base_kernel, str) and base_kernel.startswith("file:")
            else str(base_kernel)
        )
        with open(history_path, "w", encoding="utf-8") as hf:
            hf.write(
                "# bo_iter oracle_queries_batch oracle_evals_cumulative best_so_far "
                "elapsed_sec acqf_random_fallback best_kernel\n"
            )
            hf.write(
                f"bo_iter=0 oracle_queries_batch={n_init} oracle_evals_cumulative={n_init} "
                f"best_so_far={best_val(y)} elapsed_sec=0.000000 "
                f"acqf_random_fallback=False(0/0) best_kernel={kernel_label}\n"
            )

    acqf_fail_count = 0
    acqf_total_count = 0

    for it in range(num_rounds):
        remaining = n_iter - X.shape[0]
        if remaining <= 0:
            break
        q = min(batch_size, remaining)
        t_round_start = time.time()

        if verbose:
            print(f"\n[BO round {it + 1}/{num_rounds}] Fitting GP ({base_kernel})...")

        X_train = X
        y_for_model = y if maximize else -y
        Y_train = standardize(y_for_model)
        model = _initialize_model(X_train, Y_train, base_kernel, device, dtype, bounds=bounds)
        failed_acq = False
        try:
            _fit_model(model)
        except Exception:
            print("Fit failed. Falling back to random points (Sobol).")
            failed_acq = True
            # Fallback: random points if fit fails — use round number in seed to avoid duplicates
            _offset = (it + 1) * 1000 + 88888
            fallback_seed = seed + _offset if seed is not None else _offset
            sobol_fallback = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=fallback_seed)
            x_next = sobol_fallback.draw(q).to(device=device, dtype=dtype)
            for d in range(dim):
                lo, hi = bounds[d]
                x_next[:, d] = x_next[:, d] * (hi - lo) + lo
        else:
            # best_f in standardized space (model is fit on standardize(y_for_model))
            f_best = Y_train.max().item()
            X_baseline = X if acquisition == "qlognei" else None
            x_next, _ = _optimize_acquisition_with_fallback(
                model, bounds, acquisition, f_best, device, dtype, batch_size=q, X_baseline=X_baseline,
                verbose=verbose, prune_baseline=prune_baseline, seed=seed, sequential=sequential,
            )
            if x_next.dim() == 1:
                x_next = x_next.unsqueeze(0)

        y_next_val = f(x_next).reshape(-1, 1).to(device=device, dtype=dtype)
        X = torch.cat([X, x_next], dim=0)
        y = torch.cat([y, y_next_val], dim=0)

        n_before = X.shape[0] - q
        for i in range(q):
            n_evals = n_before + i + 1
            best_so_far = best_val(y[:n_evals])
            history.append({"n_evals": n_evals, "best": best_so_far, "incumbent": best_so_far})
            if on_eval:
                regret = (best_so_far - optimal_val) if optimal_val is not None else None
                on_eval(n_evals, best_so_far, regret)

        elapsed_round = time.time() - t_round_start
        acqf_total_count += 1
        if failed_acq:
            acqf_fail_count += 1
        if history_path:
            with open(history_path, "a", encoding="utf-8") as hf:
                hf.write(
                    f"bo_iter={it + 1} oracle_queries_batch={q} "
                    f"oracle_evals_cumulative={X.shape[0]} best_so_far={best_so_far} "
                    f"elapsed_sec={elapsed_round:.6f} "
                    f"acqf_random_fallback={failed_acq}({acqf_fail_count}/{acqf_total_count}) "
                    f"best_kernel={kernel_label}\n"
                )

        if verbose:
            print(f"  -> New eval: n_evals={X.shape[0]}, best={best_so_far:.4f}")

        del model
        _clear_gpu_memory()

    return X, y, history
