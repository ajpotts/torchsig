"""Demonstrate configurable pulse shaping for constellation signals.

Run from the repository root:

    python examples/scripts/demo_constellation_pulse_shaping.py

The pulse-shaping parameters are generator metadata, so they can be supplied
directly as keyword arguments. Omitting both parameters retains TorchSIG's
random pulse-shape and rolloff selection.
"""

from __future__ import annotations

from torchsig.signals.builders.constellation import ConstellationSignalGenerator

COMMON_PARAMETERS = {
    "constellation_name": "qpsk",
    "sample_rate": 10_000,
    "bandwidth_min": 1_000,
    "bandwidth_max": 1_000,
    "signal_duration_in_samples_min": 2_048,
    "signal_duration_in_samples_max": 2_048,
}


def describe(name: str, generator: ConstellationSignalGenerator) -> None:
    """Generate one signal and print its effective shaping parameters."""
    signal = generator()
    print(f"{name:18} shape={signal.pulse_shape_name:11} alpha={signal.alpha_rolloff!s:5} samples={len(signal.data)}")


def main() -> None:
    """Generate fixed and randomized pulse-shaping examples."""
    fixed_rectangular = ConstellationSignalGenerator(
        **COMMON_PARAMETERS,
        pulse_shape_name="rectangular",
        seed=1,
    )
    fixed_srrc = ConstellationSignalGenerator(
        **COMMON_PARAMETERS,
        pulse_shape_name="srrc",
        alpha_rolloff=0.35,
        seed=2,
    )
    randomized = ConstellationSignalGenerator(
        **COMMON_PARAMETERS,
        seed=3,
    )

    describe("fixed rectangular", fixed_rectangular)
    describe("fixed SRRC", fixed_srrc)
    describe("randomized", randomized)


if __name__ == "__main__":
    main()
