Plan: Decompose the frequency metadata and bandwidth fixes into three MRs

This work should be split into three MRs. The concerns have different risk
profiles, and separating them will make review and regression diagnosis easier.


MR 1: Harden canonical frequency metadata
==========================================

Purpose
-------

Finish and lock down the stale-edge fix already present in the repository.

Changes
-------

- Establish center_freq and full two-sided bandwidth as the canonical fields.
- Ensure lower_freq and upper_freq are always derived as:

      center_freq - bandwidth / 2
      center_freq + bandwidth / 2

- Exclude _lower_frequency and _upper_frequency from
  SignalMetadataObject.to_dict().
- Verify that center-frequency changes, bandwidth changes, frequency shifting,
  and boundary clipping cannot expose stale edges.
- Document clipping semantics: clipping updates both center frequency and
  bandwidth to describe the retained interval.

Tests
-----

- Updating center frequency invalidates cached edges.
- Updating bandwidth invalidates cached edges.
- Legacy cached values cannot override canonical values or enter serialized
  metadata.
- Positive- and negative-boundary clipping produces consistent, bounded
  metadata.
- Invalid or incomplete frequency metadata fails clearly.

This MR should not change bandwidth selection or measurement.


MR 2: Separate configured bandwidth from spectral measurement
==============================================================

Purpose
-------

Fix the primary source of out-of-range bandwidths.

Changes
-------

- Split update_signal_snr_bandwidth() into distinct SNR-adjustment and
  bandwidth-measurement responsibilities.
- Stop replacing canonical bandwidth with the current threshold-based estimate.
- Preserve the generator-selected bandwidth, which must remain within the
  configured per-signal or dataset range.
- If the measurement remains useful, expose it under an explicitly
  noncanonical field such as estimated_occupied_bandwidth.
- Rename or rewrite the estimator documentation. The current threshold span is
  not a 99%-power bandwidth.
- Decide whether the estimator should remain threshold-based or become a true
  integrated-power estimator. This may be deferred if it is diagnostic-only.
- Validate generator output bandwidth before placement and raise a useful error
  if a generator violates its supplied constraints.

Tests
-----

- Deterministic Wi-Fi RTS and constellation generation remains inside
  configured bounds.
- High-SNR sidelobes do not alter canonical bandwidth.
- Per-signal bandwidth overrides are honored.
- Diagnostic occupied bandwidth, if retained, does not affect placement or
  labels.
- Boundary clipping can reduce the final canonical bandwidth but cannot enlarge
  it.
- A deliberately invalid generator result is rejected.

This is the main behavior-changing MR and should receive the most focused
review.


MR 3: Unify placement and localization geometry
===============================================

Purpose
-------

Make every downstream consumer use the same full-bandwidth convention.

Changes
-------

- Fix _map_to_coordinates() to use center_freq +/- bandwidth / 2 rather than
  center_freq +/- bandwidth.
- Consider centralizing frequency-interval calculation in a helper so
  placement, clipping, validation, and labels cannot independently reinterpret
  bandwidth.
- Verify that YOLO height and center are calculated from final post-clipping
  metadata.
- Add final invariant validation before a component is inserted or labeled:

  - bandwidth is finite and positive;
  - lower_freq is less than or equal to upper_freq;
  - edges agree with center frequency and bandwidth;
  - edges are within dataset frequency limits.

- Document whether validation raises, warns, or clips. The recommended behavior
  is to clip only in the explicit placement operation and raise afterward if
  invariants are still violated.

Tests
-----

- Overlap rectangles use the full bandwidth exactly once.
- Non-overlapping signals are not rejected because of doubled rectangles.
- Cochannel overlap detection remains correct at touching and overlapping
  boundaries.
- YOLO boxes match final component edges before and after clipping.
- An end-to-end deterministic default-pipeline test covers Wi-Fi and
  constellation components.
- A dataset-wide invariant test samples enough seeded components to exercise
  varied classes and SNRs.


Merge order
===========

1. MR 1: canonical metadata
2. MR 2: bandwidth ownership
3. MR 3: placement and labels

MR 1 is foundational but low risk. MR 2 determines what bandwidth means and
which stage owns it. MR 3 then updates all geometric consumers to that finalized
contract.

These changes should not be combined into one MR. Estimator behavior changes,
metadata serialization hardening, and overlap-coordinate correction are
independently reviewable and could produce distinct downstream regressions.
Each MR can include complete tests and leave the branch in a coherent state.
