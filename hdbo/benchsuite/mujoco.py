import warnings
from typing import Any, Optional

import torch

from hdbo.benchsuite import settings
from hdbo.benchsuite.benchmark import Benchmark
from hdbo.benchsuite.utils.mujoco import func_factories


class MujocoBenchmark(Benchmark):

    def __init__(
        self,
        dim: int,
        ub: torch.Tensor,
        lb: torch.Tensor,
        benchmark: Any,
        seed: Optional[int] = None
    ):
        super().__init__(dim=dim, lb=lb, ub=ub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # If benchmark is an ObjectFactory, we pass the seed to make_object
            if hasattr(benchmark, 'make_object'):
                self.benchmark = benchmark.make_object(seed=seed)
            else:
                self.benchmark = benchmark.make_object()

    def __call__(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # Benchsuite wrapper might pass 1D or 2D tensors.
        # MuJoCo internal expects 2D numpy array (batch, dim)
        if x.ndim == 1:
            x_np = x.cpu().detach().numpy().reshape(1, -1)
        else:
            x_np = x.cpu().detach().numpy()
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # self.benchmark returns (y, None) where y is reward array
            y, _ = self.benchmark(x_np)
            
        # Return as tensor. BenchSuiteWrapper will handle further negation or item() extraction.
        # Note: MujocoBenchmark originally returned -y. We'll return y and let 
        # BenchSuiteWrapper handle maximization/negation.
        return torch.from_numpy(y).to(dtype=settings.DTYPE, device=settings.DEVICE)


class MujocoSwimmer(MujocoBenchmark):
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=16,
            ub=torch.ones(16, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(16, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["swimmer"],
            seed=seed
        )


class MujocoHumanoid(MujocoBenchmark):
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=6392,
            ub=torch.ones(6392, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(6392, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["humanoid"],
            seed=seed
        )


class MujocoAnt(MujocoBenchmark):
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=888,
            ub=torch.ones(888, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(888, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["ant"],
            seed=seed
        )


class MujocoHopper(MujocoBenchmark):
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=33,
            ub=1.4 * torch.ones(33, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1.4 * torch.ones(33, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["hopper"],
            seed=seed
        )


class MujocoWalker(MujocoBenchmark):
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=102,
            ub=0.9 * torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1.8 * torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["walker_2d"],
            seed=seed
        )


class MujocoHalfCheetah(MujocoBenchmark):
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        super().__init__(
            dim=102,
            ub=torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["half_cheetah"],
            seed=seed
        )
