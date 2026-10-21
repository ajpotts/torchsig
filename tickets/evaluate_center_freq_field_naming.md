# Evaluate a clearer name for dataset-relative center frequency

Summary

TorchSIG's `center_freq` signal metadata is a signed offset in Hz from the
containing complex-IQ capture's frequency origin. It is not normally an
absolute RF tuning frequency. The current name is concise and familiar, but it
does not communicate the reference coordinate and can be misread when working
with recorded RF data.

This ticket is an API and compatibility investigation. It must not change the
mathematical meaning of the field or frequency-placement behavior.

Questions

- Is clearer documentation sufficient, or would a name such as
  `center_freq_offset` materially prevent mistakes?
- How widely is `center_freq` used in public constructors, transforms,
  builders, labels, configuration, file formats, and downstream projects?
- Can an alias be implemented without allowing two names to contain
  inconsistent values on the same metadata object?
- How should old serialized datasets be read if a new name is introduced?
- Would a rename require a schema version and deprecation period?

Proposed work

1. Inventory public and serialized uses of `center_freq`.
2. Compare keeping the current name, adding a permanent alias, and performing
   a versioned rename.
3. Prototype alias lookup, assignment, serialization, and conflict handling if
   an alias remains a viable option.
4. Recommend one option with its compatibility and migration costs.

Acceptance criteria

- The investigation documents current API and storage dependencies.
- The recommendation explicitly addresses constructors, item and attribute
  access, transforms, target labels, YAML, and supported file handlers.
- If an alias is recommended, precedence and validation when both names are
  supplied are defined and tested in a prototype.
- Existing datasets can be read without manual metadata edits under the
  recommended migration plan.
- No frequency values or placement behavior are changed as part of this
  investigation.

Out of scope

- Adding an absolute RF tuning-frequency field.
- Correcting aggregate frequency metadata on composite wideband samples.
- Changing frequency placement, anti-alias filtering, or bandwidth semantics.
