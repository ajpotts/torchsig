# Define optional absolute RF frequency-reference metadata

Summary

TorchSIG signal `center_freq` values use a capture-relative baseband coordinate
system. Applications that ingest recorded RF data may also need the receiver's
absolute tuning frequency so they can convert a component offset into an
absolute RF center frequency.

Before adding a field such as `rf_center_freq`, TorchSIG needs a precise
contract for ownership, inheritance, transforms, serialization, and units.
Adding an underspecified field would recreate the ambiguity identified for
`center_freq`.

Questions

- Which object owns the RF reference: dataset, composite capture, component,
  or external-record metadata?
- Is the value the complex-IQ origin, receiver local-oscillator setting,
  channel center, or another physical reference?
- Do components inherit one capture-level RF reference, or may they override
  it?
- Which transforms preserve, update, or invalidate the reference?
- How should readers populate it from SigMF and other supported formats?
- How should writers represent absence and preserve backward compatibility?

Proposed work

1. Collect concrete recorded-data and synthetic-data use cases.
2. Define the field name, units, reference point, optionality, and hierarchy
   ownership.
3. Define the conversion from relative component metadata to absolute RF
   frequency, including examples near positive and negative baseband offsets.
4. Audit transforms and file handlers for propagation requirements.
5. Produce an implementation and schema-migration proposal, or document why
   the field should remain outside TorchSIG metadata.

Acceptance criteria

- The proposal distinguishes absolute RF reference from relative
  `center_freq` unambiguously.
- Ownership and parent/child inheritance behavior are specified.
- Transform preservation and invalidation rules are specified.
- SigMF and other relevant reader/writer behavior is addressed.
- Missing-value and backward-compatibility behavior are defined.
- At least one end-to-end example converts a component's relative frequency
  into an absolute RF frequency.
- No public field is added until this contract is approved.

Dependencies

- The dataset-relative `center_freq` coordinate convention must be documented
  first.
- The composite wideband parent frequency-metadata convention must be settled
  so the RF reference is attached at the correct hierarchy level.

Out of scope

- Renaming `center_freq`.
- Changing DSP frequency-placement behavior.
- Treating normalized spectrogram or YOLO coordinates as physical frequency.
