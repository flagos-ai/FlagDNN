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
The DEB build uses CUDA 12.8 / Python 3.12 / PyTorch 2.10 / Triton 3.6; the
RPM build uses CUDA 12.6 / Python 3.9 / PyTorch 2.8 / Triton 3.4.

For pre-publication validation, place the runtime and development packages from
libtriton_jit CI under `packaging/debian/local-deps/` or
`packaging/rpm/local-deps/`. These binary files are ignored by Git; normal CI
installs the same packages from FlagOS Nexus.
