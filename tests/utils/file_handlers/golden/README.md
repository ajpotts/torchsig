# HDF5 golden files

These files were written once with the first stable TorchSig HDF5 schemas and
are committed as compatibility fixtures:

- `packed_v1/data.h5`: `torchsig-packed` schema `1.0`
- `homogeneous_v1/data.h5`: `torchsig-homogeneous` schema `1`

The compatibility tests must read these files directly. Do not regenerate
them with the current writers during a test, because doing so would hide
reader incompatibilities.

SHA-256:

```text
21e8fad219fff28568f902d32d43bb9697fe3f9da95ffe05936c8bffb69ad65c  packed_v1/data.h5
5b1f37b5c491b1d03815e9fea70f23a1060136d9879d63446c1487aa86821954  homogeneous_v1/data.h5
```
