# NVIDIA / Hygon case catalog contract

This directory checks workload metadata, not numerical correctness.

`integration.hygon.dtype_catalog` compiles NVIDIA's existing native benchmark
selectors with the recording `paired.hpp` supplied here. The recorder never
builds a vendor plan or launches a kernel. It compares those selected cases
against Hygon's native adapter, including every tensor's UID, dtype, dimensions,
strides and binding offset. The probe includes TF32 cases even though production
Hygon validation explicitly skips TF32. No CUDA or cuDNN SDK is required.

The small pointwise recorder implements the dtype filter applied after the
NVIDIA selector. `pointwise-selection.sha256` guards that filter: inspect changes
to NVIDIA's selection and update both the recorder and Hygon adapter before
updating the digest. Layout and matrix selection use NVIDIA source directly.

`integration.hygon.catalog_registration` also checks benchmark registrations,
shared default selection, extended case generators and grouped report accounting.
These probes belong only to Hygon test targets. Production libraries and normal
Hygon functional/benchmark executables do not link NVIDIA validation objects.
