"""Numba-accelerated kernels for performance-critical functional transforms.

Currently provides:
  * ``sampling_clock_impairments_numba_wrapper`` - drop-in replacement for
    ``torchsig.utils.dsp.sampling_clock_impairments`` (clock_drift / clock_jitter).
    Aims for bit-identical results and fixes the RNG ordering by generating
    random numbers in the same order as NumPy.
  * ``digital_agc_numba`` - the sequential AGC sample loop used by ``digital_agc``.

Importing this module requires numba; callers should fall back to the pure-NumPy
implementations if the import fails.
"""

import numpy as np
from numba import jit
from numba.types import complex64, float32

_OUTPUT_CAPACITY_ERROR = "sampling clock output capacity exhausted"
_MAX_OUTPUT_CAPACITY_MULTIPLIER = 16


@jit(nopython=True, cache=True)
def partition_polyphase_numba(h, up_rate, taps_per_phase):
    """Numba version of partition_polyphase."""
    h_pfb = np.zeros((up_rate, taps_per_phase), dtype=np.float32)
    for phase in range(up_rate):
        tap_idx = phase
        for idx in range(taps_per_phase):
            if tap_idx < len(h):
                h_pfb[phase, idx] = h[tap_idx] * up_rate
            else:
                h_pfb[phase, idx] = 0.0
            tap_idx += up_rate
    return h_pfb


@jit(nopython=True, cache=True)
def sampling_clock_impairments_numba(
    x_real,
    x_imag,
    uprate,
    drate,
    jitter_ppm,
    drift_ppm,
    jitter_drift_pool,
    h_pfb_reversed,
    taps_per_phase,
    padded_len,
    max_input_idx,
    nominal_position_increment,
    num_output_samples,
):
    """Apply sampling-clock impairments with precomputed filter and RNG data.

    The wrapper prepares contiguous real and imaginary inputs, a reversed
    polyphase filter bank, and interleaved jitter/drift samples for this kernel.
    """
    input_padded_real = np.zeros(padded_len, dtype=np.float32)
    input_padded_imag = np.zeros(padded_len, dtype=np.float32)

    input_start = taps_per_phase - 1
    input_end = input_start + len(x_real)

    input_padded_real[input_start:input_end] = x_real
    input_padded_imag[input_start:input_end] = x_imag

    # Track the ideal sampling path in absolute polyphase units. Timing
    # impairments are accumulated separately and applied only to each read.
    nominal_position = uprate / drate
    max_sample_position = max_input_idx * uprate + (uprate - 1)

    output_real = np.zeros(num_output_samples, dtype=np.float32)
    output_imag = np.zeros(num_output_samples, dtype=np.float32)

    output_idx = 0
    while nominal_position <= max_sample_position:
        if output_idx >= num_output_samples or (output_idx * 2 + 1) >= len(jitter_drift_pool):
            raise RuntimeError(_OUTPUT_CAPACITY_ERROR)

        pool_index = output_idx * 2
        sample_position = nominal_position + jitter_drift_pool[pool_index]
        if sample_position < 0.0:
            sample_position = 0.0
        elif sample_position > max_sample_position:
            sample_position = max_sample_position

        input_idx = int(sample_position // uprate)
        phase_position = sample_position - input_idx * uprate
        phase = int(phase_position)

        acc_re = 0.0
        acc_im = 0.0

        for tap_idx in range(taps_per_phase):
            coefficient = h_pfb_reversed[phase, tap_idx]
            input_position = input_idx + tap_idx
            acc_re += coefficient * input_padded_real[input_position]
            acc_im += coefficient * input_padded_imag[input_position]

        output_real[output_idx] = acc_re
        output_imag[output_idx] = acc_im
        output_idx += 1

        nominal_position += nominal_position_increment

    result = np.zeros(output_idx, dtype=np.complex64)
    for idx in range(output_idx):
        result[idx] = output_real[idx] + 1j * output_imag[idx]

    return result


def sampling_clock_impairments_numba_wrapper(
    h,
    x,
    uprate,
    drate,
    jitter_ppm,
    drift_ppm,
    rng,
):
    """Wrapper for the numba-optimized sampling clock impairments function.

    Matches the signature of the original function and aims for bit-identical results.
    Generates interleaved jitter/drift pairs to match NumPy RNG consumption order.
    """
    if not isinstance(uprate, (int, np.integer)) or uprate <= 0:
        raise ValueError("uprate must be a positive integer")
    if not np.isfinite(drate) or drate <= 0.0:
        raise ValueError("drate must be finite and positive")
    if not np.isfinite(jitter_ppm) or jitter_ppm < 0.0:
        raise ValueError("jitter_ppm must be finite and nonnegative")
    if not np.isfinite(drift_ppm):
        raise ValueError("drift_ppm must be finite")

    nominal_position_increment = drate * (1.0 + drift_ppm * 1e-6)
    if not np.isfinite(nominal_position_increment) or nominal_position_increment <= 0.0:
        raise ValueError("drift_ppm produces a nonfinite or nonpositive sampling-position increment")

    rng = np.random.default_rng() if rng is None else rng

    # Construct the polyphase filter bank.
    taps_per_phase = int(np.ceil(len(h) / uprate))

    h_pfb = partition_polyphase_numba(h, uprate, taps_per_phase)
    h_pfb_reversed = np.ascontiguousarray(
        np.flip(h_pfb, axis=1),
        dtype=np.float32,
    )

    padded_len = len(x) + 2 * taps_per_phase - 1
    max_input_idx = padded_len - taps_per_phase

    num_output_samples = int(np.ceil(padded_len * uprate / nominal_position_increment)) + 1

    if jitter_ppm != 0.0:
        jitter_std = uprate * jitter_ppm * 1e-6

        pairs = rng.normal(0.0, 1.0, (num_output_samples, 2)).astype(np.float32)

        jitter_drift_pool = np.empty(num_output_samples * 2, dtype=np.float32)
        jitter_drift_pool[0::2] = pairs[:, 0] * jitter_std
        jitter_drift_pool[1::2] = 0.0
    else:
        jitter_drift_pool = np.zeros(num_output_samples * 2, dtype=np.float32)

    x_real = np.ascontiguousarray(x.real, dtype=np.float32)
    x_imag = np.ascontiguousarray(x.imag, dtype=np.float32)

    max_output_samples = num_output_samples * _MAX_OUTPUT_CAPACITY_MULTIPLIER

    while True:
        try:
            return sampling_clock_impairments_numba(
                x_real,
                x_imag,
                uprate,
                drate,
                jitter_ppm,
                drift_ppm,
                jitter_drift_pool,
                h_pfb_reversed,
                taps_per_phase,
                padded_len,
                max_input_idx,
                nominal_position_increment,
                num_output_samples,
            )
        except RuntimeError as exc:
            if str(exc) != _OUTPUT_CAPACITY_ERROR or num_output_samples >= max_output_samples:
                raise

        new_capacity = min(2 * num_output_samples, max_output_samples)
        additional_count = new_capacity - num_output_samples
        if jitter_ppm != 0.0:
            pairs = rng.normal(0.0, 1.0, (additional_count, 2)).astype(np.float32)
            additional_pool = np.empty(additional_count * 2, dtype=np.float32)
            additional_pool[0::2] = pairs[:, 0] * jitter_std
            additional_pool[1::2] = 0.0
        else:
            additional_pool = np.zeros(additional_count * 2, dtype=np.float32)

        jitter_drift_pool = np.concatenate((jitter_drift_pool, additional_pool))
        num_output_samples = new_capacity


@jit(nopython=True, cache=True)
def digital_agc_numba(
    data: complex64[:],  # 1D complex64 array (Numba type)
    initial_gain_db: float32,  # All scalars must be Numba types
    alpha_smooth: float32,
    alpha_track: float32,
    alpha_overflow: float32,
    alpha_acquire: float32,
    ref_level_db: float32,
    track_range_db: float32,
    low_level_db: float32,
    high_level_db: float32,
):
    """Numba version of the digital AGC sample-by-sample loop."""
    n = len(data)
    output = np.empty(n, dtype=np.complex64)  # Pre-allocate (faster than zeros_like)
    gain_db = initial_gain_db
    level_db = 0.0

    for sample_idx in range(n):
        sample = data[sample_idx]
        mag = np.abs(sample)  # MUST use np.abs (not built-in abs)
        if mag == 0.0:
            level_db = -200.0
        elif sample_idx == 0:
            level_db = np.log(mag)
        else:
            level_db = level_db * alpha_smooth + np.log(mag) * (1 - alpha_smooth)

        output_db = level_db + gain_db
        diff_db = ref_level_db - output_db

        if level_db <= low_level_db:
            alpha_adjust = 0.0
        elif output_db >= high_level_db:
            alpha_adjust = alpha_overflow
        elif np.abs(diff_db) > track_range_db:  # MUST use np.abs
            alpha_adjust = alpha_acquire
        else:
            alpha_adjust = alpha_track

        gain_db += diff_db * alpha_adjust
        output[sample_idx] = sample * np.exp(gain_db)

    return output
