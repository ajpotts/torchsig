# References for Sampling-Clock Jitter, Drift, and Fractional-Delay Resampling

These references cover three overlapping topics relevant to modeling sampling-clock impairments in a polyphase resampler:

- clock-noise and frequency-stability models;
- the effect of sampling jitter on ADC and software-radio signals; and
- fractional-delay filtering for evaluating a signal at perturbed sampling instants.

## Recommended starting references

1. **Arkesteijn, Klumperink, and Nauta, “Jitter Requirements of the Sampling Clock in Software Radio Receivers” (2006).**  
   Directly relevant to software-radio receivers. It models how the sampling-clock jitter spectrum interacts with the input-signal spectrum rather than representing jitter only by a single RMS value.  
   [Read the paper](https://ris.utwente.nl/ws/files/6732085/jitter.pdf)

2. **Löhning et al., “The Effects of Aperture Jitter and Clock Jitter in Wideband ADCs.”**  
   Discusses aperture jitter, accumulated clock jitter, and modeling clock phase as a Wiener process. It is particularly useful when distinguishing independent timing errors from errors accumulated through noisy sampling intervals.  
   [Read the paper](https://www.vodafone-chair.org/pbls/legacy/m-lohning/The_Effects_of_Aperture_Jitter_and_Clock_Jitter_in_Wideband_ADCs.pdf)

3. **W. J. Riley, _Handbook of Frequency Stability Analysis_, NIST Special Publication 1065.**  
   A practical guide to white and flicker phase noise, white and flicker frequency noise, random-walk frequency noise, deterministic drift, stability measures, and simulation of power-law clock noise. Section 8 is especially relevant to generating simulated noise.  
   [Read the NIST handbook](https://safe.nrao.edu/wiki/pub/Main/ToddHunter/nist1065.pdf)

## Foundational clock-modeling references

4. **Sullivan, Allan, Howe, and Walls, _Characterization of Clocks and Oscillators_, NIST Technical Note 1337.**  
   A broad reference on time error, fractional-frequency error, phase noise, drift, and clock-stability measures.  
   [Read NIST Technical Note 1337](https://tf.nist.gov/general/tn1337/Tn001.pdf)

5. **IEEE Std 1139-2022, _IEEE Standard Definitions of Physical Quantities for Fundamental Frequency and Time Metrology—Random Instabilities_.**  
   Useful for choosing precise parameter names and distinguishing frequency offset, time error, phase deviation, jitter, wander, and spectral-density quantities.  
   [Read IEEE 1139-2022](https://antena.fe.uni-lj.si/literatura/Razno/VFtehnika/AndrejLavric/1139-2022.pdf)

6. **Demir, Mehrotra, and Roychowdhury, “Phase Noise in Oscillators: A Unifying Theory and Numerical Methods for Characterization” (2000).**  
   A rigorous explanation of why noise in a free-running oscillator produces nonstationary phase whose variance increases with time. This provides a theoretical basis for Wiener phase models.  
   [Read the paper](https://dl.acm.org/doi/pdf/10.1145/277044.277050)

7. **Demir, “Phase Noise and Timing Jitter in Oscillators with Colored-Noise Sources” (2002).**  
   Extends oscillator timing-noise theory to colored sources, including low-frequency and \(1/f\)-type processes.  
   [Paper information and text](https://www.scribd.com/document/331803920/01159110)

8. **Lee, “Modeling Timing Jitter in Oscillators” (2001).**  
   A comparatively accessible bridge between oscillator phase noise and discrete-event timing-jitter models, including discussion of common modeling misconceptions.  
   [Read the paper](https://designers-guide.org/forum/Attachments/mentorpaper_3544.pdf)

## Fractional-delay and resampling references

9. **Välimäki and Laakso, “Principles of Fractional Delay Filters” (2000).**  
   Covers the DSP mechanism needed after generating perturbed sample times: approximating values at noninteger sample positions with fractional-delay filters.  
   [Read the paper](https://ieeexplore.ieee.org/iel5/6939/18660/00860248.pdf)

10. **Välimäki and Laakso, “Fractional Delay Filters—Design and Applications.”**  
    A fuller treatment connecting fractional-delay filters with nonuniform sampling and sample-rate conversion.  
    [Read the chapter](https://link.springer.com/content/pdf/10.1007/978-1-4615-1229-5_20.pdf)

## Suggested reading order

1. Arkesteijn et al. for the software-radio problem.
2. Löhning et al. for ADC clock and aperture jitter.
3. NIST SP 1065 for a taxonomy of clock-noise processes and their simulation.
4. Välimäki and Laakso for the fractional-delay implementation.
5. Demir et al. for rigorous oscillator phase-noise theory.

## Connection to the current implementation

The implementation under discussion approximately follows

\[
D_n = D_{n-1} + W_n,
\]

\[
q_{n+1} = q_n + d + J_n + D_n.
\]

In clock-metrology terms:

- `q_step` represents accumulated sampling position or phase.
- `drate + ...` represents the sampling-position increment and therefore corresponds to clock rate or sampling period.
- `clock_jitter`, because it is added to every increment, behaves more like white frequency noise than independent sampling-time jitter.
- `clock_drift` is a random walk in frequency.
- Adding `clock_drift` repeatedly to `q_step` integrates that random-walk frequency into time error.
- `drift_ppm` therefore describes the magnitude of a random frequency increment per output sample, not an ordinary fixed PPM clock offset.

A conventional fixed sampling-clock offset would instead use a constant fractional rate error, while independent aperture jitter would perturb individual sampling instants without accumulating from one instant to the next. The references above provide the terminology and mathematical models needed to decide which behavior TorchSig should implement.
