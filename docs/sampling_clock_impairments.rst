Sampling-clock impairments
==========================

TorchSig separates independent aperture jitter from time-varying sampling-
clock drift. Both effects use an arbitrary-phase polyphase interpolator and
preserve the input array length.

Clock drift
-----------

``ClockDrift`` supports three rate-error trajectories:

* ``linear`` interpolates from ``initial_drift_ppm`` to ``drift_ppm`` across
  the capture.
* ``random_walk`` accumulates Gaussian rate changes. ``abs(drift_ppm)`` is the
  RMS endpoint scale.
* ``filtered_noise`` produces correlated Gaussian rate error.
  ``abs(drift_ppm)`` is its stationary RMS scale and
  ``filtered_noise_alpha`` controls correlation.

For a physically parameterized linear trajectory, provide
``drift_rate_ppm_per_second`` and ``sample_rate``. In that mode the
instantaneous error is

.. math::

   d[k] = d_0 + r k / f_s,

where :math:`d_0` is ``initial_drift_ppm``, :math:`r` is the drift rate in
PPM/second, and :math:`f_s` is the sample rate in samples/second.

Linear drift sampling positions are evaluated from the closed-form integral
of this rate error. They do not repeatedly add floating-point position
increments, so numerical error does not accumulate along long captures.

Clock jitter
------------

``ClockJitter`` applies independent, zero-mean Gaussian displacement to each
sampling instant. ``jitter_ppm`` is the RMS timing displacement in millionths
of one input-sample period. The displacement does not accumulate into later
samples.

Alignment and boundaries
------------------------

The default initial phase is zero, so a zero-magnitude impairment is aligned
with the input. A nonzero ``initial_phase`` intentionally applies a fractional-
sample phase.

If positive drift exhausts the available input before producing the requested
output length, ``boundary_mode`` selects the policy:

* ``edge`` repeats the final available sample and is the default.
* ``zeros`` pads with complex zeros.
* ``wrap`` repeats the available output periodically.
* ``raise`` rejects the operation.

Bandwidth limitation
--------------------

The prototype filter is intended for signals comfortably inside Nyquist.
Signals occupying the band edge can experience attenuation or image leakage
that is larger than a small PPM-scale clock impairment. Validate the response
for an application's occupied bandwidth before using nearly full-band input.
