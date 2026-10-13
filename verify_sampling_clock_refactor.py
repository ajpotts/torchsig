#!/usr/bin/env python3
"""Compare current sampling_clock_impairments with the version at a Git ref.

Run from the TorchSig repository root:
    python /path/to/verify_sampling_clock_refactor.py --base HEAD^

Use --base <commit-or-branch> to select the actual pre-refactor revision.
The script extracts only the old function from Git and executes it using the
current torchsig.utils.dsp module's dependencies; it does not modify the repo.
"""
from __future__ import annotations

import argparse
import ast
import copy
import subprocess

import numpy as np
from torchsig.utils import dsp


def load_original(ref: str):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:torchsig/utils/dsp.py"], text=True
    )
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "sampling_clock_impairments"]
    if len(matches) != 1:
        raise RuntimeError("Could not locate exactly one original function")
    func = copy.deepcopy(matches[0])
    func.decorator_list = []
    namespace = dict(vars(dsp))
    exec(compile(ast.Module(body=[func], type_ignores=[]), f"{ref}:dsp.py", "exec"), namespace)
    return namespace["sampling_clock_impairments"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD^", help="Git ref before the refactor")
    args = parser.parse_args()
    original = load_original(args.base)
    current = dsp.sampling_clock_impairments

    cases = [
        ("no impairment", 0., 0., 1, 1., 0., 16, 1),
        ("jitter only", 100., 0., 4, 3., 0.25, 128, 37),
        ("drift only", 0., 100., 4, 3., 0.25, 128, 37),
        ("jitter and drift", 100., 100., 4, 3., 0.25, 128, 37),
        ("downsampling", 25., 10., 8, 11., 0.5, 256, 65),
        ("upsampling", 10., 20., 8, 3., 0.1, 128, 33),
        ("empty output", 0., 0., 2, 1000., 0., 4, 9),
        ("short filter", 50., 50., 1, 1., 0., 32, 1),
    ]

    for name, jitter, drift, up, down, phase, length, taps in cases:
        signal_rng = np.random.default_rng(1001)
        x = (signal_rng.standard_normal(length) + 1j * signal_rng.standard_normal(length)).astype(np.complex64)
        h = signal_rng.standard_normal(taps).astype(np.float32)
        kwargs = dict(h=h, x=x, uprate=up, drate=down, jitter_ppm=jitter,
                      drift_ppm=drift, initial_phase=phase)
        old_rng = np.random.default_rng(2026)
        new_rng = np.random.default_rng(2026)
        old = original(**kwargs, rng=old_rng)
        new = current(**kwargs, rng=new_rng)
        np.testing.assert_array_equal(new, old, err_msg=f"Output differs: {name}")
        assert new.dtype == old.dtype, f"Dtype differs: {name}"
        assert old_rng.bit_generator.state == new_rng.bit_generator.state, (
            f"RNG state differs: {name}"
        )
        print(f"PASS {name:20s}  output samples={len(new):5d}, exact IQ and RNG state")

    print(f"\nAll {len(cases)} comparisons passed against {args.base}.")


if __name__ == "__main__":
    main()
