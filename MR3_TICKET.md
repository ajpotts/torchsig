# Unify component placement and localization frequency geometry

## Summary

TorchSIG documents component `bandwidth` as the full two-sided occupied width,
with frequency edges at `center_freq - bandwidth / 2` and
`center_freq + bandwidth / 2`. Wideband overlap placement does not follow that
contract: `_map_to_coordinates()` uses `center_freq +/- bandwidth` and
normalizes frequency by half the sample rate. This expands and mis-scales the
rectangle used for cochannel-overlap decisions.

YOLO labels use the full-bandwidth convention, but they do not explicitly
validate that their source interval is finite, positive, mutually consistent,
and within dataset frequency bounds. Placement and labeling therefore lack a
shared final-frequency invariant.

## Impact

- Overlap rectangles can cover substantially more spectrum than their signal.
- Valid non-overlapping components may be rejected as overlapping.
- Placement and YOLO labels can interpret the same bandwidth differently.
- Invalid or out-of-bounds frequency metadata may reach localization labels
  without a targeted error.

## Requested change

- Add one canonical frequency-interval validator to signal metadata.
- Validate final component metadata after frequency shifting and clipping.
- Map overlap rectangles from the derived `lower_freq` and `upper_freq` values.
- Normalize frequency edges by the full sample rate when converting them to FFT
  coordinates.
- Use inclusive axis-aligned interval comparisons for overlap detection so
  partial, collinear, and touching rectangles are handled consistently.
- Validate frequency metadata before generating YOLO labels.
- Ensure YOLO labels use final post-clipping center frequency and bandwidth.

## Acceptance criteria

- The canonical interval is always `center_freq +/- bandwidth / 2`.
- Center frequency and bandwidth must be finite, and bandwidth must be positive.
- Optional dataset frequency bounds must be finite, ordered, and inclusive.
- Final generated components outside dataset frequency bounds are rejected.
- Overlap rectangles map the canonical lower and upper edges exactly once.
- Separated intervals do not overlap, while touching and intersecting intervals
  do.
- YOLO height equals `bandwidth / sample_rate` and its center uses the final
  component center frequency.
- A component clipped at either dataset boundary produces a valid YOLO box.
- An out-of-bounds component cannot be labeled.
- Deterministic tests cover interval validation, overlap mapping, clipping, and
  localization labels.
