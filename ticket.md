Background
TorchSig currently randomizes many signal-generation parameters, such as the pulse-shaping rolloff (alpha) for QPSK. There is no convenient way to override these parameters for a specific signal class—for example, to generate every QPSK signal with a fixed rolloff.
More broadly, experiment configuration is difficult to reproduce and compare when generator settings must be specified programmatically. A YAML configuration file would allow configurations to be version controlled, reviewed, and diffed between experiments.
Requested change
Add support for optional YAML-based signal-generation configuration, including per-signal-class parameter overrides.
For example:
signals: qpsk: parameters: alpha: value: 0.35
The configuration system should eventually be able to represent any supported TorchSig configuration change, while allowing unspecified settings to retain their existing defaults and randomization behavior.
Initial scope


        
      Load an optional experiment configuration from YAML.

        
      Validate signal class names, parameter names, value types, and allowed ranges.

        
      Allow generator parameters to be overridden for an individual signal class.

        
      Support fixed values initially, including a fixed QPSK pulse-shaping rolloff.

        
      Preserve the existing randomized behavior when a parameter is not overridden.

        
      Expose the resolved configuration in generated-dataset metadata or another experiment artifact.

        
      Produce clear errors for unknown or invalid settings.

        
      Maintain backward compatibility for users who do not provide a YAML file.

Future extensions
The schema should be designed so it can later support:


        
      Parameter ranges or probability distributions

        
      Shared defaults across signal classes

        
      Dataset, impairment, transform, and target configuration

        
      Reproducibility settings such as random seeds

        
      Configuration inheritance or reusable presets

        
      Serialization of the complete resolved experiment configuration

A possible future syntax could be:
`signals:
qpsk:
parameters:
alpha:
distribution: uniform
min: 0.2
max: 0.5
16qam:
parameters:
alpha:
choices: [0.2, 0.35, 0.5]`
Acceptance criteria


        
      A user can supply a YAML file that fixes QPSK alpha to a chosen value.

        
      All generated QPSK examples use that value.

        
      Other QPSK parameters and other signal classes retain their current behavior unless configured.

        
      Invalid classes, parameters, or values fail before dataset generation with actionable messages.

        
      The effective configuration is recorded for reproducibility.

        
      Existing APIs and default dataset-generation behavior remain unchanged without a configuration file.

        
      Tests cover YAML parsing, validation, fixed-value application, fallback behavior, and malformed configurations.

        
      User documentation includes the schema and at least one complete example.
 
