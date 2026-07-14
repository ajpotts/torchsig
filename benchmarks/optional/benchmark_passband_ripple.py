import numpy as np
import pytest

from torchsig.transforms.functional import passband_ripple

# Constants
_EPS = np.finfo(np.float32).eps
TLL = 10 * np.log10(_EPS)  # "Ten Log Lynx" - a safe lower bound for 20*log10(x)

def _safe_log10(x):
    """A log10 that clamps its input to [EPS, inf) to avoid -inf."""
    x = np.maximum(x, _EPS)
    return np.log10(x)

# Benchmark test cases
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("num_taps","max_ripple_db","coefficient_decay_rate","signal_length"),
    [
        (3, 2.0, 1.0, 1000),       # Small filter, short signal
        (10, 1.0, 0.5, 10000),     # Medium filter, medium signal
        (32, 0.5, 1.0, 100000),    # Large filter, long signal
        (50, 0.1, 2.0, 50000),     # Very selective filter
    ],
)
def test_passband_ripple_benchmark(
    benchmark, num_taps, max_ripple_db, coefficient_decay_rate, signal_length
):
    """Benchmark the passband_ripple function with various configurations."""
    # Generate random test data
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, signal_length) + 1j * rng.normal(0, 1, signal_length)

    # Run the benchmark
    result = benchmark(
        passband_ripple,
        data,
        num_taps=num_taps,
        max_ripple_db=max_ripple_db,
        coefficient_decay_rate=coefficient_decay_rate,
        rng=rng,
    )

    # Basic sanity checks on the result
    assert result.shape == data.shape
    assert np.iscomplexobj(result)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

# Additional benchmark for the worst-case scenario (when filter can't be found)
@pytest.mark.benchmark
def test_passband_ripple_worst_case_benchmark(benchmark):
    """Benchmark the worst-case scenario where filter can't be found."""
    # Use parameters that are very hard to satisfy
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000) + 1j * rng.normal(0, 1, 1000)

    result = benchmark(
        passband_ripple,
        data,
        num_taps=50,
        max_ripple_db=0.01,  # Very strict requirement
        coefficient_decay_rate=0.1,
        max_counter=100,     # Limited attempts
        fallback="original",
        rng=rng,
    )

    # In this case, it should return the original signal
    assert np.array_equal(result, data.astype(np.complex64))
