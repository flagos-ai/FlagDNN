%global debug_package %{nil}

# PyTorch and CUDA are supplied by the NVIDIA runtime rather than RPMs.
# Keep automatic requirements for distro libraries, Python, and TritonJIT.
%global __requires_exclude ^(libcuda[.]so[.]1|libtorch(_cpu|_cuda|_python)?[.]so|libc10(_cuda)?[.]so)[(][)][(]64bit[)]$

Name:           libflagdnn-nvidia
Version:        0.2.0
Release:        2%{?dist}
Summary:        FlagDNN native graph runtime (NVIDIA backend)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagDNN
Source0:        flag-dnn-%{version}.tar.gz

BuildRequires:  cmake >= 3.23
BuildRequires:  gcc-c++
BuildRequires:  libtriton-jit-nvidia-devel >= 0.1.0-3
BuildRequires:  ninja-build
BuildRequires:  patchelf
BuildRequires:  python3-devel
BuildRequires:  python3-pyyaml
Requires:       libtriton-jit-nvidia%{?_isa} >= 0.1.0-3
Requires:       python3
Requires:       python3-pyyaml

%description
FlagDNN provides a native C/C++ graph runtime and a dynamically loaded NVIDIA
backend using CUDA and the system Triton JIT runtime. This package contains the
shared libraries, compiler providers, kernels, registries, and tuning data.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       libtriton-jit-nvidia-devel%{?_isa} >= 0.1.0-3

%description devel
C/C++ headers and CMake package files for applications using FlagDNN.

%prep
%autosetup -n flag-dnn-%{version}

%build
PY3_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=/usr/local/lib/python${PY3_VER}/site-packages:/usr/local/lib64/python${PY3_VER}/site-packages:$(python3 -c "import site; print(':'.join(site.getsitepackages()))")
export PATH=/usr/local/bin:$PATH
NVIDIA_LIBRARY_PATH=$(find /usr/local/lib/python${PY3_VER}/site-packages/nvidia -type d -name lib -printf '%%p:' 2>/dev/null)
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:-}"
%cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
    -DFLAGDNN_BACKENDS=nvidia \
    -DFLAGDNN_BUILD_BENCHMARKS=OFF \
    -DFLAGDNN_BUILD_TESTS=OFF \
    -DFLAGDNN_CODEGEN_PYTHON=/usr/bin/python3 \
    -DFLAGDNN_EXECUTION_ENGINE=libtriton_jit
%cmake_build

%install
%cmake_install
find %{buildroot}%{_libdir} -name '*.so*' -type f \
    -exec patchelf --remove-rpath {} \;

%check
test -f %{buildroot}%{_libdir}/libflagdnn.so.1
test -f %{buildroot}%{_libdir}/libflagdnn_backend_nvidia.so.2
test -f %{buildroot}%{_libdir}/cmake/FlagDNN/FlagDNNConfig.cmake
test -f %{buildroot}%{_datadir}/flagdnn/kernels/registry.json

%files
%license LICENSE
%doc README.md
%{_libdir}/libflagdnn.so.[0-9]*
%{_libdir}/libflagdnn_backend_nvidia.so.2*
%{_datadir}/flagdnn/

%files devel
%{_includedir}/flagdnn/
%{_includedir}/flagdnn_frontend.h
%{_libdir}/libflagdnn.so
%{_libdir}/libflagdnn_backend_nvidia.so
%{_libdir}/cmake/FlagDNN/

%changelog
* Fri Aug 07 2026 FlagOS Contributors <contact@flagos.io> - 0.2.0-2
- Replace the retired Python package with native runtime packages
- Build the NVIDIA backend against system libtriton-jit-nvidia
- Ship the complete installed SDK resources and CMake package files
- Preserve distro requirements while filtering NVIDIA/PyTorch SONAMEs
- Follow the upstream 0.2.0 core library ABI

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging
