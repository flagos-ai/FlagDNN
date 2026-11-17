#!/bin/bash
set -euo pipefail

# Match the libtriton-jit build stack for each distribution.
. /etc/os-release
json_package=json-devel
torch_index=()
case "${ID}" in
    rocky|rhel|almalinux|centos)
        dnf install -y epel-release dnf-plugins-core
        dnf config-manager --set-enabled crb 2>/dev/null || \
            dnf config-manager --set-enabled powertools
        python_packages='torch==2.8.0 triton==3.4.0 pybind11==3.1.0'
        suite=rhel9
        ;;
    openEuler)
        json_package=nlohmann-json-devel
        python_packages='torch==2.8.0+cpu triton==3.4.0 pybind11==3.1.0'
        torch_index=(--extra-index-url https://download.pytorch.org/whl/cpu)
        suite=openeuler24.03
        printf '%%dist .oe%s\n' "${VERSION_ID//./}" > /etc/rpm/macros.flagdnn-dist
        ;;
    fedora)
        # Fedora 43's Python 3.14 needs the newer libtriton-jit stack.
        python_packages='torch==2.9.0+cpu triton==3.5.0 pybind11==3.1.0'
        torch_index=(--extra-index-url https://download.pytorch.org/whl/cpu)
        suite=fedora43
        curl -fsSL -o /etc/yum.repos.d/cuda-fedora43.repo \
            https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo
        dnf install -y cuda-nvcc-13-2 cuda-cudart-devel-13-2 \
            cuda-driver-devel-13-2 cuda-cccl-13-2
        ;;
    *)
        echo "Unsupported RPM build distribution: ${ID}" >&2
        exit 1
        ;;
esac

dnf install -y --allowerasing cmake fmt-devel gcc-c++ git "${json_package}" \
    ninja-build patchelf python3-devel python3-pip python3-packaging \
    python3-pyyaml rpm-build rpmdevtools findutils tar gzip
dnf clean all
read -r -a packages <<< "${python_packages}"
python3 -m pip install --no-cache-dir "${torch_index[@]}" "${packages[@]}"

# The fallback must use this distro's repository, not an EL9 package on another ABI.
printf '[flagos]\nname=FlagOS\nbaseurl=%s\nenabled=1\ngpgcheck=0\n' \
    "${FLAGOS_YUM:-https://resource.flagos.net/repository/flagos-yum-${suite}}" \
    > /etc/yum.repos.d/flagos.repo
