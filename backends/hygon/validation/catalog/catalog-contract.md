# NVIDIA / Hygon case catalog contract

This directory checks workload metadata, not numerical correctness.

`integration.hygon.dtype_catalog` compiles NVIDIA's existing native benchmark
selectors with the recording `paired.hpp` supplied here. The recorder never
builds a vendor plan or launches a kernel. It compares those selected cases
against Hygon's native adapter, including every tensor's UID, dtype, dimensions,
strides and binding offset. Both the probe and production Hygon validation
include explicit IEEE and TF32 cases. No CUDA or cuDNN SDK is required for this
metadata probe.

The small pointwise recorder mirrors selection after the NVIDIA factories.
The current NVIDIA runner executes every selected case without a dtype filter.
`pointwise-selection.sha256` guards this selection policy: inspect changes
to NVIDIA's selection and update the recorder and Hygon adapter as needed before
updating the digest. Layout and matrix selection use NVIDIA source directly.

`integration.hygon.catalog_registration` also checks benchmark registrations,
shared default selection, extended case generators and grouped report accounting.
These probes belong only to Hygon test targets. Production libraries and normal
Hygon functional/benchmark executables do not link NVIDIA validation objects.
