"""
High-dimensional blackbox optimization benchmarks.
Real-world problems for evaluating BO in high dimensions.

HDBO_BENCHMARKS is loaded lazily so `import hdbo` does not require torch until accessed.
"""

from typing import Any

__all__ = ["HDBO_BENCHMARKS"]


def __getattr__(name: str) -> Any:
    if name == "HDBO_BENCHMARKS":
        from .benchsuite import HDBO_BENCHMARKS

        return HDBO_BENCHMARKS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
