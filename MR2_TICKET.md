# Preserve configured component bandwidth during wideband generation

## Summary

Wideband generation currently replaces each generator-selected component
`bandwidth` with a value derived from a max-hold spectrogram. The estimator
takes the span between the first and last FFT bins more than 3 dB above the
configured noise floor and describes that span as a 99%-power bandwidth.

This is not an integrated-power measurement. At high SNR, sidelobes, spectral
leakage, and isolated bins can make the estimated span several times wider than
the configured component bandwidth. The result becomes the canonical bandwidth
used by placement and labels, so generated datasets no longer honor their
configured bandwidth distribution.

## Impact

- Protocol and constellation components can exceed `bandwidth_max`.
- Frequency placement may unnecessarily invoke boundary clipping.
- Localization metadata and labels can describe leakage rather than the
  generator-selected channel width.
- Dataset bandwidth distributions can differ substantially from configuration.

## Requested change

- Separate SNR adjustment from spectral bandwidth estimation.
- Preserve the generator-selected, full two-sided `bandwidth` as canonical
  metadata during normal generation.
- Retain the existing threshold-span estimate only as optional diagnostic
  metadata, with a name and documentation that do not imply 99%-power
  measurement semantics.
- Validate that generators which declare configurable bandwidth bounds return a
  finite, positive value within those bounds.
- Explicitly document tone as a fixed 1 Hz metadata-width exception that does
  not sample the configured bandwidth distribution.
- Preserve the existing `update_signal_snr_bandwidth()` public entry point for
  compatibility.

Placement rectangle and YOLO geometry changes are outside this ticket and are
reserved for MR3 in `ticket_plan.md`.

## Acceptance criteria

- SNR adjustment does not modify canonical `bandwidth`.
- The threshold-span estimate is stored separately as
  `estimated_occupied_bandwidth` when it is available.
- The estimate is documented as threshold-based rather than a 99%-power
  bandwidth.
- Configurable generators cannot return non-finite, non-positive, or
  out-of-range bandwidths without an error.
- Tone accepts exactly its documented fixed 1 Hz metadata width.
- Deterministic Wi-Fi RTS and constellation components retain bandwidths within
  the configured range even when their spectral estimates are wider.
- Default wideband generation produces no out-of-range non-tone component
  bandwidths.
