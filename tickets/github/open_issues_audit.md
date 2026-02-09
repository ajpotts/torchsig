# GitHub Open-Issue Audit

Snapshot of the 12 open issues in [TorchDSP/torchsig](https://github.com/TorchDSP/torchsig/issues?q=is%3Aissue%20state%3Aopen), fetched on 2026-09-21 and assessed against the code on the local `tickets` branch.

“Addressed” means the requested behavior or fix is present in this checkout. It does not mean the GitHub issue has been closed, released, or fully accepted upstream. Open pull requests are excluded. Effort estimates include implementation, focused tests, and review:

- **XS:** less than half a day
- **S:** about 0.5–2 days
- **M:** about 3–5 days
- **L:** about 1–2 weeks
- **XL:** more than 2 weeks

“Critical” is **Yes** only where the issue can silently corrupt datasets/labels or reliably break a primary, default workflow. Accuracy defects confined to an optional impairment, documentation gaps, and feature requests are not classified as critical.

| Ticket | Addressed in this checkout? | Effort | Critical bug? | Assessment |
|---|---|---:|:---:|---|
| [#487 — `iq_imbalance` applies uniform gain](https://github.com/TorchDSP/torchsig/issues/487) | **No** | S | No | `functional.iq_imbalance` still multiplies I and Q by the same `10 ** (amplitude_imbalance / 10)` factor, so it changes overall gain rather than relative channel amplitude. |
| [#482 — PassbandRipple rejection budget crashes valid inputs](https://github.com/TorchDSP/torchsig/issues/482) | **Yes — in 2.2.0, not 2.1.x** | M | **Yes** | TorchSig 2.1.0 and 2.1.1 retain the failing 1,000-attempt rejection loop. The replacement first appears in the 2.2.0 development commit `9452092`: `passband_ripple` was redesigned to enforce at least 65 odd taps, return the original input by default on design failure, and offer `fallback="raise"`. Focused shape, gain, and fallback tests were added. This also addresses #473. |
| [#480 — Spectrogram lacks x/y support vectors](https://github.com/TorchDSP/torchsig/issues/480) | **No** | M | No | `Spectrogram` replaces `signal.data` with the STFT result but does not attach time/frequency coordinate vectors or persist them as self-contained HDF5 metadata. |
| [#478 — HDF5 parent metadata key reuse corrupts metadata](https://github.com/TorchDSP/torchsig/issues/478) | **No** | S | **Yes** | `_assign_hdf5_keys` gives stable counter keys to signals and component signals, but not to their metadata parent chains; parents still fall back to `str(id(obj))`. The reported reuse/collision path therefore remains. |
| [#476 — Clock impairment normalization causes alignment jumps](https://github.com/TorchDSP/torchsig/issues/476) | **No** | L | No | `clock_drift` and `clock_jitter` still center-crop/pad solely from output length. The odd-discard path can still produce `slice_back == 0` followed by `[:-0]`, and sampling origin/group delay is not tracked. |
| [#475 — Clarify/validate clock-jitter stochastic model](https://github.com/TorchDSP/torchsig/issues/475) | **No** | M | No | The implementation still adds independent jitter to the accumulated `q_step`, while docs describe Gaussian sampling-phase effects without governing equations, variance growth, or statistical validation. Resolving intended semantics requires a design decision before coding. |
| [#473 — Passband ripple breaks impairment levels 1 and 2](https://github.com/TorchDSP/torchsig/issues/473) | **Yes — in 2.2.0, not 2.1.x** | M | **Yes** | Same root cause as #482. The bug is present in 2.1.0 and 2.1.1. Commit `9452092` identifies itself as version 2.2.0 and introduces the redesigned `passband_ripple`, which no longer raises by default when design fails and has focused fallback tests. |
| [#472 — Stale/inverted frequency bounds in generated HDF5](https://github.com/TorchDSP/torchsig/issues/472) | **Yes** | M | **Yes** | `SignalMetadataObject` now treats `center_freq` and `bandwidth` as canonical, invalidates legacy cached edges on mutation, and derives `lower_freq`/`upper_freq` dynamically. Tests cover invalidation and derived bounds. |
| [#471 — Ambiguous two-sided FM bandwidth/Nyquist check](https://github.com/TorchDSP/torchsig/issues/471) | **Yes** | XS | No | `fm_modulator` now explicitly documents a full, two-sided 3 dB bandwidth spanning approximately `[-bandwidth/2, bandwidth/2]` and validates `bandwidth <= sample_rate`. |
| [#283 — GUI for dataset generation](https://github.com/TorchDSP/torchsig/issues/283) | **No** | XL | No | No GUI framework or dataset-configuration UI is present. The existing YAML/script workflow is not the requested graphical editor and generator. |
| [#282 — WAV, SigMF, and OGG file-format support](https://github.com/TorchDSP/torchsig/issues/282) | **Yes** | L | No | `WAVReader`, `OGGReader`, and `SigMFReader` are implemented and exported; each maps recordings to complex `Signal` data, with tests including SigMF and file-reader behavior. OGG intentionally supports lossless streams only. |
| [#280 — Transform and signal-builder contribution guide](https://github.com/TorchDSP/torchsig/issues/280) | **Yes** | M | No | `docs/contributing_transforms_and_signal_builders.rst` contains setup, architecture, implementation, registration, documentation, and testing guidance for both transforms and builders, and is included in the docs index. |

## Summary

- **6 addressed:** #482, #473, #472, #471, #282, #280
- **6 not addressed:** #487, #480, #478, #476, #475, #283
- **4 critical bugs:** #482 and #473 (duplicate reports of the same default-pipeline failure), #478 (silent metadata corruption), and #472 (incorrect persisted frequency labels)
- **Highest-priority unresolved issue:** #478, because it can silently corrupt labels in ordinary multi-worker dataset generation. Its proposed fix is small and directly targets the remaining `id()` fallback on transient parent metadata.

## Audit limitations

This is a source-level audit of the current checkout, supplemented by existing focused tests. It does not claim that addressed changes have shipped in a published release, and it does not treat an open pull request as code already addressed in this checkout.
