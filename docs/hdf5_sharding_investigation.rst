Structured HDF5 Sharding Investigation
======================================

Decision
--------

MR13 recommends **no-go for MR14** with the evidence currently available. The
quick MR12 IQ run reached 46.5% of the device-only ceiling, but that gap does
not isolate single-file contention. Moving from zero to two workers improved
shuffled materialized throughput from 17,862.9 to 22,609.1 samples/s (1.27x),
which is evidence against a contention plateau at the tested worker counts.
The test used uncompressed data, so decompression was not responsible. Only one
repetition and the full layout were measured, preventing a stable causal claim.

The observed gap can still include storage bandwidth, HDF5 selection cost,
collation, host-to-device transfer, and model work. The independent loader and
end-to-end timings are too short and noisy to allocate the gap reliably. Adding
a sharded format now would introduce lifecycle and operational complexity
without demonstrated benefit.

Reproducing the gate
--------------------

Analyze one or more MR12 JSON artifacts with:

.. code-block:: console

   python benchmarks/analyze_stage2_bottlenecks.py \
       .benchmarks/stage2-quick-results.json \
       --output .benchmarks/stage2-quick-bottlenecks.json

The gate requires at least three repetitions with at most 5% rate standard
deviation. It first requires a material multi-worker regression consistent with
single-file contention, then a matched sharded prototype that improves the
single-file result by at least 10%. Without both signals, MR14 remains closed.

Prospective manifest design
---------------------------

If future evidence opens MR14, use one versioned JSON manifest as the dataset
entry point. It should contain:

* format name and version, schema fingerprint, total sample count, and creation
  configuration;
* an ordered shard table with relative path, global half-open index range,
  sample count, byte size, and cryptographic checksum;
* the same structured leaf schema and validation metadata used by unsharded
  files; and
* completion state plus a generation identifier so incomplete publications are
  rejected.

Global index lookup should binary-search shard end offsets, subtract the
selected shard's start offset, and preserve requested order and duplicates.
Batch reads should group indices by shard, use the existing per-shard coalesced
reader, and restore sampler order. Each DataLoader worker should lazily keep a
small least-recently-used set of process-local handles; the configured cap must
stay below the operating system file-descriptor budget. Sampler semantics must
remain globally shuffled rather than shuffling only within shards.

Publication, validation, and errors
-----------------------------------

Write all shards and a draft manifest into a unique sibling staging directory.
Validate schema, counts, ranges, sizes, checksums, and sampled/full values before
marking the manifest complete. Atomically rename the directory only after every
shard is durable; overwrite should retain and restore the previous generation
on failure. Readers must fail before yielding samples when the manifest is
missing, incomplete, overlapping, non-contiguous, or references a missing,
truncated, schema-mismatched, or checksum-invalid shard. Error messages should
identify the shard and global index range.

Append and resume should not mutate a published generation. A future append
operation should create a new generation that references verified immutable
old shards plus new staged shards, then atomically publish a new manifest.
Interrupted creation may resume only from a private staging generation after
revalidating every completed shard.

Operational tradeoffs and estimate
----------------------------------

Shard size should be selected by bytes as well as samples: large enough to
avoid excessive manifests, opens, and tiny I/O, but small enough for practical
copy/recovery and worker locality. Evaluation must include shuffle quality,
file-descriptor limits, cold and warm cache behavior, local and network storage,
and representative total dataset sizes.

If the gate later passes, MR14 is estimated at roughly 8--12 engineering days:
manifest/schema and atomic writer (3--4), indexed reader and handle cache
(3--4), corruption/multiprocessing/shuffle tests (2--3), and migration,
benchmarking, and documentation (1--2). This estimate excludes remote object
storage and distributed writer support.
