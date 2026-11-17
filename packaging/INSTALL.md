# Installing FlagDNN packages

FlagDNN is a native C/C++ graph runtime. The retired `python3-flag-dnn`
package is no longer produced.

- `libflagdnn-nvidia` contains the core runtime, NVIDIA backend plugin,
  compiler providers, kernels, registries, and tuning data.
- `libflagdnn-nvidia-dev` contains the public headers and CMake package files.

The NVIDIA runtime consumes `libtriton-jit-nvidia` from the FlagOS package
repository. PyTorch, Triton, Python, and CUDA must match the matrix used to
build that runtime. GPU-enabled PyTorch and Triton remain vendor runtime
dependencies and are not replaced with a CPU-only distro PyTorch package.
The DEB build uses CUDA 12.8 / Python 3.12 / PyTorch 2.10 / Triton 3.6.

| RPM distribution | Base image | Python | Build PyTorch | Triton |
| --- | --- | --- | --- | --- |
| EL9 | `nvidia/cuda:12.6.0-devel-rockylinux9` | 3.9 | 2.8.0 | 3.4.0 |
| openEuler 24.03 | `openeuler/cuda:13.0.0-oe2403lts` | 3.11 | 2.8.0+cpu | 3.4.0 |
| Fedora 43 | `fedora:43` with CUDA 13.2 | 3.14 | 2.9.0+cpu | 3.5.0 |

Pass the base image as the first argument to `packaging/rpm/build-flag-dnn-rpm.sh`.
The CPU torch wheels provide build-time C++ libraries on the two new distributions.
They do not replace the NVIDIA driver and matching GPU runtime on the target.
The fallback package repository follows the selected distribution.
Direct Docker builds can override it with the `FLAGOS_YUM` build argument.

For pre-publication validation, place the runtime and development packages from
libtriton_jit CI under `packaging/debian/local-deps/` or
`packaging/rpm/local-deps/`. These binary files are ignored by Git; normal CI
installs the same packages from FlagOS Nexus.
Use packages from the same distribution and Python ABI as the build image.
The Fedora/openEuler builds need local packages until libtriton-jit is published
to those repositories. Installation and CPU consumer checks do not validate GPU execution.
