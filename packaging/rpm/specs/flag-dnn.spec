%global debug_package %{nil}

# Filter the auto-generated Requires for: torch.
# Reason: distro torch is CPU-only; users install GPU torch via pip.
# See packaging/INSTALL.md (or future flagos-packaging install docs) for the
# user-side pip install incantation.
%global __requires_exclude ^python3.*dist.*(torch)

Name:           python3-flag-dnn
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagDNN — DNN primitives for FlagOS

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagDNN
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flag-dnn-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 60
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  pyproject-rpm-macros

%description
Deep neural network operators (convolutions, normalizations, activations) targeting FlagOS-supported accelerators via Triton kernels.

%prep
%autosetup -n flag-dnn-%{version}

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files flag_dnn

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_dnn'); assert s and s.origin, 'flag_dnn not findable'; print('OK: flag_dnn at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE

%changelog
* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging.
