# Prevent stale frequency-edge metadata from being serialized

## Summary

`SignalMetadataObject` treats `center_freq` and full two-sided `bandwidth` as
the canonical frequency metadata. `lower_freq` and `upper_freq` are derived
from those fields, but legacy `_lower_frequency` and `_upper_frequency` values
can still be present in an object's local metadata and included by `to_dict()`.

Although current property access prefers the canonical fields and updates to
`center_freq` or `bandwidth` invalidate cached edges, serializing legacy edge
values exposes two representations of the same interval. A completed metadata
snapshot can therefore contain stale or contradictory frequency bounds.

## Impact

- Persisted metadata may contain internal frequency-edge fields that disagree
  with the canonical center frequency and bandwidth.
- Consumers that read the serialized internal fields directly may construct
  incorrect localization labels or frequency bounds.
- Debugging output can appear inconsistent even when the runtime properties
  return the correct values.

## Requested change

- Keep `center_freq` and full two-sided `bandwidth` as the canonical serialized
  frequency fields.
- Continue deriving `lower_freq` and `upper_freq` as
  `center_freq - bandwidth / 2` and `center_freq + bandwidth / 2`.
- Exclude `_lower_frequency` and `_upper_frequency` from `to_dict()` output.
- Document that boundary clipping updates both canonical fields to describe the
  retained interval.
- Cover positive- and negative-boundary clipping and serialization with
  deterministic tests.

This ticket does not change generator bandwidth selection, spectral bandwidth
measurement, placement geometry, or YOLO label calculations. Those changes are
tracked in the subsequent MRs described in `ticket_plan.md`.

## Acceptance criteria

- Updating `center_freq` or `bandwidth` cannot leave an active stale edge cache.
- Runtime lower and upper edges agree with the canonical fields.
- `_lower_frequency` and `_upper_frequency` are absent from serialized metadata.
- Boundary clipping produces mutually consistent center frequency, bandwidth,
  lower edge, and upper edge values.
- Final clipped edges remain within the configured dataset frequency bounds.
- Existing metadata and file-format tests continue to pass.
