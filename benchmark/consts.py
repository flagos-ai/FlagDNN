"""Benchmark constants and result models.

This module mirrors the FlagGems benchmark layout while reusing the existing
FlagDNN definitions from attri_util. New benchmark files should import consts
instead of attri_util directly.
"""

from benchmark.attri_util import (  # noqa: F401
    ALL_AVAILABLE_METRICS,
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_ITER_COUNT,
    DEFAULT_METRICS,
    DEFAULT_SHAPES,
    DEFAULT_WARMUP_COUNT,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchMode,
    OperationAttribute,
    check_metric_dependencies,
    get_recommended_shapes,
    model_shapes,
)
