"""Frequency-response contract for the sampling-clock prototype filter."""

import numpy as np
from scipy import signal as sp

from torchsig.utils.dsp import SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH, prototype_polyphase_filter


def test_sampling_clock_prototype_meets_documented_bandwidth_contract() -> None:
    uprate = 5000
    taps = prototype_polyphase_filter(uprate)
    passband_edge = SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH / 2
    frequencies = np.array([0.0, passband_edge, 0.7])

    _, response = sp.freqz(taps, worN=2 * np.pi * frequencies / uprate)
    response_db = 20 * np.log10(np.abs(response) / np.abs(response[0]))

    assert abs(response_db[1]) < 0.05
    assert response_db[2] < -80.0


def test_sampling_clock_bandwidth_contract_is_full_two_sided_bandwidth() -> None:
    assert SAMPLING_CLOCK_MAX_OCCUPIED_BANDWIDTH == 2 * 0.35
