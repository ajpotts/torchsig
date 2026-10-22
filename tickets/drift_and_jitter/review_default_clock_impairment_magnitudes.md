Title: Review default clock impairment magnitudes and sampling policy

Priority: Low

Summary
-------

The functional defaults use 10 ppm, and `ClockJitter` samples from 1 to 10 ppm.
For typical short captures these settings can produce effects far below common
channel-noise levels. The transforms may therefore add computational cost
without a learnable or visibly measurable impairment in common dataset
configurations.

This is a usability and dataset-design question rather than a correctness bug;
defaults should not be changed without measuring representative workloads and
considering reproducibility and compatibility.

Proposed investigation
----------------------

* Quantify impairment power and accumulated timing error across representative
  capture lengths, occupied bandwidths, SNRs, and modulation families.
* Compare current defaults with realistic hardware ranges and deliberately
  augmented training ranges.
* Decide whether to retain the defaults, revise transform distributions, or
  provide documented physical and augmentation-oriented presets.
* Clearly distinguish ppm frequency/period error from instantaneous timing
  jitter in user-facing guidance.

Acceptance criteria
-------------------

* The decision is supported by reproducible measurements over representative
  dataset configurations.
* Documentation explains when the default effect is expected to be observable.
* Any changed default is called out as a compatibility-affecting change and is
  covered by deterministic distribution and transform tests.
* If defaults remain unchanged, recommended ranges or presets are documented
  for users who need materially observable augmentation.
