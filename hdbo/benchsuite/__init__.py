import torch
import numpy as np
from .svm import SVM as SVMClass
from .mopta08 import Mopta08 as Mopta08Class
from .rover import RoverBenchmark as RoverClass
from .mujoco import MujocoHumanoid
from .lasso import LassoDNA

class BenchSuiteWrapper:
    def __init__(self, benchmark_class, maximize=False, seed=None):
        self._benchmark_class = benchmark_class
        self._maximize = maximize
        self._seed = seed
        # Deferred initialization to avoid overhead during registry creation
        self._instance = None

        # We need dim and bounds now for the registry
        # We pass seed=None for the temp instance used for registry info
        temp_instance = benchmark_class()
        self.dim = temp_instance.dim
        self.bounds = list(zip(temp_instance.lb.cpu().tolist(), temp_instance.ub.cpu().tolist()))

    @property
    def instance(self):
        if self._instance is None:
            try:
                self._instance = self._benchmark_class(seed=self._seed)
            except TypeError:
                # Fallback for benchmarks that don't support seed arg yet
                self._instance = self._benchmark_class()
                if self._seed is not None and hasattr(self._instance, 'seed'):
                    try:
                        self._instance.seed(self._seed)
                    except:
                        pass
        return self._instance

    def set_seed(self, seed):
        self._seed = seed
        if self._instance is not None:
            if hasattr(self._instance, 'seed'):
                try:
                    self._instance.seed(seed)
                except:
                    pass
            # For MujocoPolicyFunc, we might need a more direct way if already initialized
            if hasattr(self._instance, 'benchmark') and hasattr(self._instance.benchmark, '_seed'):
                self._instance.benchmark._seed = seed

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the benchmark on a batch of points X.
        X shape: (..., D)
        Return shape: (...)
        """
        *batch_shape, d = X.shape
        X_flat = X.view(-1, d)

        # All provided benchmarks (SVM, Mopta08, Rover, MuJoCo, Lasso)
        # have been updated to handle batches efficiently.
        vals = self.instance(X_flat)

        # Ensure vals is a tensor and correct shape
        if not isinstance(vals, torch.Tensor):
            vals = torch.tensor(vals, dtype=X.dtype, device=X.device)

        # Reshape to 1D batch if needed
        if vals.dim() > 1:
            vals = vals.reshape(-1)

        return vals.view(*batch_shape)

def _make_benchmark_dict(benchmark_class, maximize=False):
    wrapper = BenchSuiteWrapper(benchmark_class, maximize=maximize)
    return {
        "f": wrapper,
        "dim": wrapper.dim,
        "bounds": wrapper.bounds,
        "maximize": maximize,
        "n_init_default": 20,
    }

# Only the 5 benchmarks used in the paper main table.
HDBO_BENCHMARKS = {
    "SVM_388":   _make_benchmark_dict(SVMClass,        maximize=False),
    "Mopta08":   _make_benchmark_dict(Mopta08Class,    maximize=False),
    "Rover":     _make_benchmark_dict(RoverClass,      maximize=True),
    "Humanoid":  _make_benchmark_dict(MujocoHumanoid,  maximize=True),
    "Lasso-DNA": _make_benchmark_dict(LassoDNA,        maximize=False),
}
