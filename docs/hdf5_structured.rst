Structured HDF5 Materialization
===============================

Structured HDF5 stores fixed-schema, model-ready samples without reconstructing
TorchSig ``Signal`` objects on every training epoch. Use it to precompute work
that should remain identical across epochs, such as deterministic transforms,
representation conversion, and target construction.

Supported sample grammar
------------------------

A sample may be a numeric NumPy array, PyTorch tensor, or scalar, or a nested
combination of tuples, lists, string-keyed mappings, and fixed-shape numeric or
boolean leaves. Every sample must have the same container structure, mapping
keys, leaf shapes, and NumPy dtypes. Complex, floating-point, integer, and
boolean dtypes are preserved.

Object arrays, strings, empty containers, variable-sized leaves, ragged
detection targets, and changing mapping keys are not supported. Pad or encode
variable-sized targets into a fixed-shape representation before materializing.

Materializing a Dataset
-----------------------

:func:`torchsig.datasets.materialize_structured_dataset` returns a map-style
:class:`torchsig.datasets.StructuredHDF5Dataset` that can be used immediately.
The default Dataset path uses an identity collator, preserving each sample's
tuple, list, and mapping containers.

.. code-block:: python

   from torch.utils.data import DataLoader
   from torchsig.datasets import materialize_structured_dataset

   materialized = materialize_structured_dataset(
       training_dataset,
       "/data/model-ready/train",
       batch_size=64,
       num_workers=4,
       overwrite=False,
       validation="sampled",
       validation_samples=256,
       validation_seed=0,
   )

   loader = DataLoader(
       materialized,
       batch_size=64,
       shuffle=True,
       num_workers=4,
       persistent_workers=True,
   )

   for inputs, targets in loader:
       train_step(inputs, targets)

   materialized.close()

``StructuredHDF5Dataset`` implements single-index access and routes DataLoader
index batches through ``StructuredHDF5Reader.read_indices``. Contiguous
requests use one slice per schema leaf. Shuffled requests preserve sampler
order and duplicate indices; adjacent sorted runs are coalesced when that
materially reduces HDF5 operations. Sparse requests retain direct reads because
h5py point selections and unnecessary batch copies are slower for that case.
Each worker lazily opens its own HDF5 handle; forked handles are reopened after
the process changes, while spawn serialization excludes HDF5 objects. Explicit
``close()`` is safe, and later access reopens the file.

Materializing an existing DataLoader
------------------------------------

An existing DataLoader is consumed with its configured sampler, workers, and
collator. Collated numeric leaves are split along their leading dimension.
Identity-collated lists of complete samples are also supported.

.. code-block:: python

   materialized = materialize_structured_dataset(
       configured_loader,
       "/data/model-ready/train",
       expected_length=len(configured_loader.dataset),
       overwrite=True,
       validation="full",
   )

For an unsized iterable dataset, ``expected_length`` is required. A mismatch
between the expected and emitted sample counts fails materialization.

Deterministic and randomized transforms
---------------------------------------

Materialize expensive operations whose output should not vary by epoch:

* deterministic filtering, FFTs, spectrograms, or representation conversion;
* deterministic normalization; and
* fixed-shape class, regression, or detection target construction.

Keep randomized training augmentation online when it is intended to change
across epochs. Materializing random crops, noise, fading, or other stochastic
augmentation freezes the particular outputs observed during materialization.
A common pattern is to materialize the deterministic base representation and
wrap the returned dataset with lightweight randomized transforms.

Atomic publication and overwrite
--------------------------------

Materialization writes to a unique sibling staging directory. The staged file
is completed and physically validated before it is renamed into place.
``overwrite=False`` rejects a non-empty existing destination.
``overwrite=True`` keeps the existing destination until the replacement is
ready and restores it if publication fails. Expected write and validation
failures clean up staging artifacts and do not remove a valid published
dataset.

Validation
----------

Validation is optional. ``validation="full"`` compares every output with the
exact sample emitted by the source stream. ``validation="sampled"`` compares
a reproducible subset selected by ``validation_samples`` and
``validation_seed``. Comparisons check containers, keys, shapes, dtypes, and
values. Exact equality is the default; NaNs compare equal and complex values
are supported. Set ``validation_rtol`` or ``validation_atol`` only for an
explicitly lossy workflow.

Validation runs before atomic publication. Successful files record the mode,
result, checked count, seed, and tolerances in HDF5 attributes. Validation
errors identify the sample index and field path.

Compatibility and migration
---------------------------

Structured files use the separate ``torchsig-structured-homogeneous`` format.
They are not interchangeable with legacy, packed, or Signal-oriented
homogeneous HDF5 files. Those formats remain supported and can serve as the
source Dataset for a one-time structured materialization pass. Existing files
are not accelerated in place; training must use the returned structured
dataset or open the structured copy explicitly.

Conversion has an up-front compute and storage cost. It is most useful when it
removes Python object construction, metadata decoding, deterministic transforms,
or general-purpose collation that would otherwise repeat every epoch. Measure
your workload rather than assuming a speedup.

Performance benchmark
---------------------

The optional benchmark compares a deterministic online FFT-based transform
with materialized structured reads. It reports materialization time, output
size, cold first-batch latency, steady epoch throughput, and the break-even
epoch count for sequential and shuffled access with zero and multiple workers.

.. code-block:: console

   python benchmarks/optional/benchmark_structured_hdf5.py \
       --samples 1024 --workers 0 2 4 --access sequential shuffled \
       --compression none --chunk-samples 32

Use representative sample sizes and worker counts. Compare compression modes:
a smaller LZF file may be slower when CPU decompression or shuffled chunk reads
dominate. Break-even is reported only when materialized epochs are faster than
online epochs.
