"""Compiler identity and dependency closure for the mthreads provider."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import sys
from typing import Any

from flagdnn_codegen.kernel_registry import (
    iter_kernel_registry_sources,
    resolve_kernel_source,
    select_kernel_candidate,
)

from .compiler_graph import TARGET_PATTERN, validate_target
from .environment_identity import identity_sha256 as environment_identity_sha256


GRAPH_SCHEMA_VERSION = 3
ARTIFACT_SCHEMA_VERSION = 1
EXECUTION_PROGRAM_VERSION = 1
PROVIDER_NAME = "mthreads_triton"
PROVIDER_VERSION = "1"
SUPPORTED_ENGINE = "libtriton_jit"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_MODULES = (
    "compiler.py",
    "compiler_graph.py",
    "compiler_identity.py",
    "compiler_tensor.py",
    "environment_identity.py",
    "execution_plan.py",
)
_IDENTITY_RESOURCES = (
    "mcc",
    "musa_driver",
    "musa_driver_header",
    "musa_runtime",
    "musa_runtime_header",
    "triton_jit",
    "triton_jit_config",
    "triton_jit_gen_ssig",
    "triton_jit_header",
    "triton_jit_standalone_compile",
)
_PYTHON_MODULES = ("torch", "torch_musa", "triton", "yaml")
_INSTALL_METADATA_NAME = "flagdnn_mthreads_install.json"
_INSTALLED_PROVENANCE_ONLY_RESOURCES = {
    "triton_jit_config",
    "triton_jit_header",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"environment JSON key is duplicated: {key!r}"
            )
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > (1 << 20):
        raise ValueError(f"mthreads environment report is invalid: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(
                    f"environment JSON nonfinite number is forbidden: "
                    f"{constant}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read mthreads environment report: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("mthreads environment report must be an object")
    return value


def _environment_report_path() -> Path | None:
    explicit_environment = os.environ.get(
        "FLAGDNN_MTHREADS_ENVIRONMENT_REPORT", ""
    )
    if explicit_environment:
        return Path(explicit_environment).expanduser().resolve(strict=True)
    adjacent = Path(__file__).resolve().with_name(
        "flagdnn_mthreads_environment.json"
    )
    if adjacent.is_file():
        return adjacent
    return None


def _file_record(
    value: object,
    context: str,
) -> tuple[Path, str, str | None]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} record is missing")
    path_value = value.get("realpath")
    digest = value.get("sha256")
    size = value.get("size")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise ValueError(f"{context} file record is invalid")
    path = Path(path_value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{context} file disappeared: {path}") from error
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{context} realpath is not canonical")
    if resolved.stat().st_size != size or _sha256_file(resolved) != digest:
        raise ValueError(f"{context} bytes differ from environment identity")
    version = value.get("version")
    if version is not None and not isinstance(version, str):
        raise ValueError(f"{context} version is invalid")
    return resolved, digest, version


def _installed_metadata_path(
    environment_source: Path | None,
) -> Path | None:
    if environment_source is None:
        return None
    provider_directory = Path(__file__).resolve().parent
    adjacent_environment = provider_directory / environment_source.name
    if environment_source != adjacent_environment:
        return None
    metadata = provider_directory / _INSTALL_METADATA_NAME
    if metadata.is_file():
        return metadata.resolve(strict=True)

    # A copied provider fixture without a FlagDNN library is allowed to model
    # the historical source-layout identity contract.  A real SDK, however,
    # must never silently fall back to the system TritonJIT when its relocation
    # metadata is missing.
    if len(provider_directory.parents) >= 4:
        sdk = provider_directory.parents[3]
        installed_core = tuple(sdk.glob("lib*/libflagdnn.so*"))
        if installed_core:
            raise ValueError(
                "installed mthreads compiler relocation metadata is missing"
            )
    return None


def _installed_file_record(
    *,
    provider_directory: Path,
    sdk: Path,
    value: object,
    context: str,
) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {
        "relative_path",
        "sha256",
    }:
        raise ValueError(f"{context} install record is invalid")
    relative_path = value.get("relative_path")
    expected_digest = value.get("sha256")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
        or not isinstance(expected_digest, str)
        or _SHA256_PATTERN.fullmatch(expected_digest) is None
    ):
        raise ValueError(f"{context} install record is invalid")
    try:
        path = (provider_directory / relative_path).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{context} installed file is missing") from error
    if not path.is_file() or not path.is_relative_to(sdk):
        raise ValueError(f"{context} installed path escapes the FlagDNN SDK")
    if _sha256_file(path) != expected_digest:
        raise ValueError(f"{context} installed bytes differ")
    return path, expected_digest


def _relocate_installed_environment(
    environment_source: Path,
    value: dict[str, Any],
) -> Path | None:
    metadata_path = _installed_metadata_path(environment_source)
    if metadata_path is None:
        return None
    metadata = _read_json(metadata_path)
    if set(metadata) != {
        "backend",
        "environment_identity_sha256",
        "libtriton_jit",
        "schema_version",
        "scripts",
    }:
        raise ValueError("mthreads install metadata root keys are invalid")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("backend") != "mthreads"
        or metadata.get("environment_identity_sha256")
        != value.get("identity_sha256")
    ):
        raise ValueError("mthreads install metadata identity is invalid")

    provider_directory = Path(__file__).resolve().parent
    if len(provider_directory.parents) < 4:
        raise ValueError("installed mthreads provider path is malformed")
    sdk = provider_directory.parents[3].resolve(strict=True)
    if not metadata_path.is_relative_to(sdk):
        raise ValueError("mthreads install metadata escapes its SDK")

    resources = value.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("mthreads install resources are invalid")
    original_jit = resources.get("triton_jit")
    jit_record = metadata.get("libtriton_jit")
    if (
        not isinstance(original_jit, dict)
        or not isinstance(jit_record, dict)
        or set(jit_record) != {
            "provenance_sha256",
            "relative_path",
            "sha256",
            "soname",
        }
        or jit_record.get("provenance_sha256")
        != original_jit.get("sha256")
        or jit_record.get("soname") != original_jit.get("soname")
    ):
        raise ValueError("installed mthreads TritonJIT metadata is invalid")
    jit_path, jit_digest = _installed_file_record(
        provider_directory=provider_directory,
        sdk=sdk,
        value={
            "relative_path": jit_record.get("relative_path"),
            "sha256": jit_record.get("sha256"),
        },
        context="libtriton_jit",
    )
    if (
        jit_path.name != jit_record.get("soname")
        or jit_path.parent.name != "mthreads"
        or jit_path.parent.parent.name != "flagdnn"
    ):
        raise ValueError("installed mthreads TritonJIT path is not private")

    scripts = metadata.get("scripts")
    if not isinstance(scripts, dict) or set(scripts) != {
        "gen_ssig.py",
        "standalone_compile.py",
    }:
        raise ValueError("installed mthreads JIT script metadata is invalid")
    script_resources = {
        "gen_ssig.py": "triton_jit_gen_ssig",
        "standalone_compile.py": "triton_jit_standalone_compile",
    }
    installed_scripts: dict[str, tuple[Path, str]] = {}
    for filename, resource_name in script_resources.items():
        script_record = scripts.get(filename)
        if (
            not isinstance(script_record, dict)
            or set(script_record)
            != {"provenance_sha256", "relative_path", "sha256"}
        ):
            raise ValueError(
                f"installed mthreads {filename} metadata is invalid"
            )
        script_path, script_digest = _installed_file_record(
            provider_directory=provider_directory,
            sdk=sdk,
            value={
                "relative_path": script_record.get("relative_path"),
                "sha256": script_record.get("sha256"),
            },
            context=filename,
        )
        original_script = resources.get(resource_name)
        provenance_digest = script_record.get("provenance_sha256")
        if (
            script_path.name != filename
            or script_path.parent.name != "scripts"
            or not isinstance(original_script, dict)
            or not isinstance(provenance_digest, str)
            or provenance_digest != original_script.get("sha256")
            or (
                filename == "gen_ssig.py"
                and script_digest != provenance_digest
            )
        ):
            raise ValueError(
                f"installed mthreads {filename} provenance is invalid"
            )
        installed_scripts[resource_name] = (script_path, script_digest)

    relocated_jit = dict(original_jit)
    relocated_jit["realpath"] = str(jit_path)
    relocated_jit["requested_path"] = str(jit_path)
    relocated_jit["sha256"] = jit_digest
    relocated_jit["size"] = jit_path.stat().st_size
    relocated_jit["runpath"] = ["$ORIGIN"]
    resources["triton_jit"] = relocated_jit
    for resource_name, (script_path, script_digest) in installed_scripts.items():
        original_script = resources[resource_name]
        relocated_script = dict(original_script)
        relocated_script["realpath"] = str(script_path)
        relocated_script["requested_path"] = str(script_path)
        relocated_script["sha256"] = script_digest
        relocated_script["size"] = script_path.stat().st_size
        resources[resource_name] = relocated_script
    return metadata_path


def _identity_resource_names(
    environment_source: Path | None,
) -> tuple[str, ...]:
    if _installed_metadata_path(environment_source) is None:
        return _IDENTITY_RESOURCES
    return tuple(
        name
        for name in _IDENTITY_RESOURCES
        if name not in _INSTALLED_PROVENANCE_ONLY_RESOURCES
    )


@lru_cache(maxsize=1)
def _validated_environment() -> tuple[Path | None, dict[str, Any]]:
    source = _environment_report_path()
    if source is None:
        from .environment_identity import collect_environment, _finalize

        value = _finalize(collect_environment())
    else:
        value = _read_json(source)
    expected_keys = {
        "schema_version",
        "platform",
        "resources",
        "python",
        "errors",
        "identity_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("mthreads environment report root keys are invalid")
    if (
        value.get("schema_version") != 2
        or value.get("platform") != "mthreads"
        or value.get("errors") != []
    ):
        raise ValueError("mthreads environment discovery did not pass")
    declared_identity = value.get("identity_sha256")
    payload = dict(value)
    payload.pop("identity_sha256", None)
    if (
        not isinstance(declared_identity, str)
        or _SHA256_PATTERN.fullmatch(declared_identity) is None
        or environment_identity_sha256(value) != declared_identity
    ):
        raise ValueError("mthreads environment identity is invalid")

    install_metadata = (
        _relocate_installed_environment(source, value)
        if source is not None
        else None
    )
    resources = value.get("resources")
    python = value.get("python")
    if (
        not isinstance(resources, dict)
        or not isinstance(python, dict)
    ):
        raise ValueError("mthreads environment sections are invalid")
    jit = resources.get("triton_jit")
    if (
        not isinstance(jit, dict)
        or jit.get("backend") != "MUSA"
        or jit.get("soname") != "libtriton_jit.so"
    ):
        raise ValueError("mthreads selected a non-MUSA TritonJIT")
    for name in _identity_resource_names(source):
        _file_record(resources.get(name), f"resources.{name}")
    if install_metadata is not None and not install_metadata.is_file():
        raise ValueError("mthreads install metadata disappeared")

    executable = python.get("executable")
    executable_path, _, _ = _file_record(
        executable, "python.executable"
    )
    if Path(sys.executable).resolve() != executable_path:
        raise RuntimeError(
            "active compiler Python differs from environment identity"
        )
    modules = python.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("mthreads Python module records are invalid")
    for name in _PYTHON_MODULES:
        _file_record(modules.get(name), f"python.modules.{name}")
        record = modules[name]
        if not isinstance(record.get("version"), str):
            raise ValueError(
                f"python.modules.{name}.version is invalid"
            )
    return source, value


def _compiler_entry_path() -> Path:
    import flagdnn_codegen

    return Path(flagdnn_codegen.__file__).resolve().with_name("main.py")


def _identity_inputs(
    compiler_entry: Path,
    environment_source: Path | None,
    environment: dict[str, Any],
) -> dict[str, Path]:
    provider_directory = Path(__file__).resolve().parent
    inputs: dict[str, Path] = {
        f"provider:{name}": (provider_directory / name).resolve()
        for name in _LOCAL_MODULES
    }
    inputs["codegen:main.py"] = compiler_entry.resolve()
    for name in ("kernel_registry.py", "provider_loader.py"):
        path = compiler_entry.with_name(name)
        if not path.is_file():
            raise ValueError(f"common compiler module is missing: {path}")
        inputs[f"codegen:{name}"] = path.resolve()
    resource_root = compiler_entry.parents[2]
    for path in iter_kernel_registry_sources("mthreads"):
        label = path.resolve().relative_to(resource_root).as_posix()
        inputs[f"registry:{label}"] = path.resolve()

    binary_candidate = select_kernel_candidate("mthreads", "add")
    if binary_candidate.ownership != "platform":
        raise ValueError("mthreads Add must resolve to platform ownership")
    if {
        "binary_contiguous_kernel",
        "binary_strided_kernel",
    }.difference(binary_candidate.functions):
        raise ValueError("mthreads Add candidate is missing required functions")
    inputs["kernel:mthreads:binary.py"] = resolve_kernel_source(
        compiler_entry, binary_candidate
    ).resolve()
    unary_candidate = select_kernel_candidate("mthreads", "relu")
    if unary_candidate.ownership != "platform":
        raise ValueError("mthreads ReLU must resolve to platform ownership")
    if {
        "unary_pointwise_contiguous_kernel",
        "unary_pointwise_strided_kernel",
    }.difference(unary_candidate.functions):
        raise ValueError(
            "mthreads unary candidate is missing required functions"
        )
    inputs["kernel:mthreads:unary.py"] = resolve_kernel_source(
        compiler_entry, unary_candidate
    ).resolve()
    identity_candidate = select_kernel_candidate("mthreads", "identity")
    if identity_candidate.ownership != "platform":
        raise ValueError("mthreads Identity must resolve to platform ownership")
    if {
        "identity_contiguous_packed_kernel",
        "identity_contiguous_kernel",
        "identity_strided_kernel",
    }.difference(identity_candidate.functions):
        raise ValueError(
            "mthreads Identity candidate is missing required functions"
        )
    inputs["kernel:mthreads:identity.py"] = resolve_kernel_source(
        compiler_entry, identity_candidate
    ).resolve()
    ternary_candidate = select_kernel_candidate(
        "mthreads", "binary_select"
    )
    if ternary_candidate.ownership != "common":
        raise ValueError(
            "mthreads binary_select must resolve to common ownership"
        )
    if {
        "binary_select_tensor_kernel",
        "binary_select_strided_kernel",
    }.difference(ternary_candidate.functions):
        raise ValueError(
            "common ternary candidate is missing required functions"
        )
    inputs["kernel:common:ternary.py"] = resolve_kernel_source(
        compiler_entry, ternary_candidate
    ).resolve()
    reshape_candidate = select_kernel_candidate("mthreads", "reshape")
    if reshape_candidate.ownership != "platform":
        raise ValueError("mthreads Reshape must resolve to platform ownership")
    if {"reshape_contiguous_kernel", "layout_copy_kernel"}.difference(
        reshape_candidate.functions
    ):
        raise ValueError(
            "mthreads Reshape candidate is missing required functions"
        )
    inputs["kernel:mthreads:layout.py"] = resolve_kernel_source(
        compiler_entry, reshape_candidate
    ).resolve()
    slice_candidate = select_kernel_candidate("mthreads", "slice")
    if slice_candidate.ownership != "platform" or {
        "slice_copy_kernel",
        "layout_copy_kernel",
    }.difference(slice_candidate.functions):
        raise ValueError("mthreads Slice candidate is incomplete")
    transpose_candidate = select_kernel_candidate("mthreads", "transpose")
    if transpose_candidate.ownership != "platform" or {
        "transpose_physical_copy_kernel",
        "layout_copy_kernel",
    }.difference(transpose_candidate.functions):
        raise ValueError("mthreads Transpose candidate is incomplete")
    if (
        resolve_kernel_source(compiler_entry, slice_candidate).resolve()
        != inputs["kernel:mthreads:layout.py"]
        or resolve_kernel_source(
            compiler_entry, transpose_candidate
        ).resolve()
        != inputs["kernel:mthreads:layout.py"]
    ):
        raise ValueError("mthreads layout candidates do not share one source")
    reduction_candidate = select_kernel_candidate(
        "mthreads", "reduction_sum"
    )
    if reduction_candidate.ownership != "common":
        raise ValueError(
            "mthreads reduction must resolve to common ownership"
        )
    if {
        "reduction_2d_kernel",
        "reduction_3d_kernel",
        "reduction_strided_kernel",
    }.difference(reduction_candidate.functions):
        raise ValueError(
            "common reduction candidate is missing required functions"
        )
    inputs["kernel:common:reduction.py"] = resolve_kernel_source(
        compiler_entry, reduction_candidate
    ).resolve()
    matmul_candidate = select_kernel_candidate("mthreads", "matmul")
    if matmul_candidate.ownership != "platform":
        raise ValueError("mthreads Matmul must resolve to platform ownership")
    if {
        "matmul_descriptor_kernel",
        "_matmul_tle_consumer",
        "_matmul_tle_producer",
        "matmul_tle_kernel",
        "matmul_strided_kernel",
    }.difference(matmul_candidate.functions):
        raise ValueError(
            "mthreads Matmul candidate is missing required functions"
        )
    inputs["kernel:mthreads:matmul.py"] = resolve_kernel_source(
        compiler_entry, matmul_candidate
    ).resolve()
    convolution_candidate = select_kernel_candidate(
        "mthreads", "convolution_fprop"
    )
    if (
        convolution_candidate.ownership != "platform"
        or convolution_candidate.source != "convolution.py"
    ):
        raise ValueError(
            "mthreads Fprop must resolve to the platform convolution kernel"
        )
    if {
        "conv1d_gemm_kernel",
        "conv2d_spatial_nchw_kernel",
        "conv3d_spatial_ncdhw_m_kernel",
        "_conv_fprop2d_im2col_kernel",
        "_conv_fprop2d_im2col_mm_kernel",
        "conv_dgrad_nd_kernel",
        "conv_wgrad_nd_kernel",
    }.difference(convolution_candidate.functions):
        raise ValueError(
            "mthreads Fprop candidate is missing required functions"
        )
    fprop_convolution_source = resolve_kernel_source(
        compiler_entry, convolution_candidate
    ).resolve()
    normalization_requirements = {
        "layernorm": {"layer_norm_kernel"},
        "rmsnorm": {"rms_norm_kernel"},
        "batchnorm": {
            "batch_norm_nchw_kernel",
            "batch_norm_kernel",
        },
    }
    normalization_source: Path | None = None
    for operation, functions in normalization_requirements.items():
        candidate = select_kernel_candidate("mthreads", operation)
        if (
            candidate.ownership != "common"
            or candidate.source != "normalization.py"
            or functions.difference(candidate.functions)
        ):
            raise ValueError(
                f"mthreads {operation} candidate is invalid"
            )
        source = resolve_kernel_source(compiler_entry, candidate).resolve()
        if normalization_source is not None and source != normalization_source:
            raise ValueError(
                "mthreads normalization operations resolve to different sources"
            )
        normalization_source = source
    if normalization_source is None:
        raise ValueError("mthreads normalization source is missing")
    inputs["kernel:common:normalization.py"] = normalization_source
    attention_requirements = {
        "sdpa": {"_sdpa_fwd_kernel"},
        "sdpa_backward": {
            "_zero_contiguous_kernel",
            "_sdpa_bwd_dq_dbias_kernel",
            "_sdpa_bwd_dkdv_kernel",
            "_sdpa_bwd_dk_kernel",
            "_sdpa_bwd_dv_kernel",
        },
        "sdpa_fp8": {
            "_zero_sdpa_fp8_fwd_amax_kernel",
            "_sdpa_fp8_fwd_kernel",
        },
        "sdpa_fp8_backward": {
            "_zero_sdpa_fp8_bwd_amax_kernel",
            "_sdpa_fp8_bwd_dq_kernel",
            "_sdpa_fp8_bwd_dkdv_kernel",
        },
    }
    attention_source: Path | None = None
    for operation, functions in attention_requirements.items():
        candidate = select_kernel_candidate("mthreads", operation)
        if (
            candidate.ownership != "common"
            or candidate.source != "attention.py"
            or functions.difference(candidate.functions)
        ):
            raise ValueError(
                f"mthreads {operation} candidate is invalid"
            )
        source = resolve_kernel_source(compiler_entry, candidate).resolve()
        if attention_source is not None and source != attention_source:
            raise ValueError(
                "mthreads Attention operations resolve to different sources"
            )
        attention_source = source
    if attention_source is None:
        raise ValueError("mthreads Attention source is missing")
    inputs["kernel:common:attention.py"] = attention_source
    inference_candidate = select_kernel_candidate(
        "mthreads", "batchnorm_inference"
    )
    if (
        inference_candidate.ownership != "platform"
        or inference_candidate.source != "normalization.py"
        or {
            "batch_norm_inference_nchw_kernel",
            "batch_norm_inference_kernel",
        }.difference(inference_candidate.functions)
    ):
        raise ValueError(
            "mthreads batchnorm_inference platform candidate is invalid"
        )
    inputs["kernel:platform:normalization.py"] = resolve_kernel_source(
        compiler_entry, inference_candidate
    ).resolve()
    dgrad_candidate = select_kernel_candidate(
        "mthreads", "convolution_dgrad"
    )
    required_dgrad_functions = {
        "conv_dgrad_nd_kernel",
        "_conv_dgrad2d_dense_pack_filter_kernel",
        "_conv_dgrad2d_dense_pack_loss_kernel",
        "_conv_dgrad2d_dense_mm_kernel",
    }
    if (
        dgrad_candidate.ownership != "platform"
        or dgrad_candidate.source != "convolution.py"
        or not required_dgrad_functions.issubset(dgrad_candidate.functions)
    ):
        raise ValueError(
            "mthreads Dgrad must resolve to the platform convolution kernel"
        )
    platform_convolution_source = resolve_kernel_source(
        compiler_entry, dgrad_candidate
    ).resolve()
    if platform_convolution_source != fprop_convolution_source:
        raise ValueError(
            "mthreads Fprop and Dgrad resolve to different sources"
        )
    wgrad_candidate = select_kernel_candidate(
        "mthreads", "convolution_wgrad"
    )
    if wgrad_candidate.ownership != "platform":
        raise ValueError(
            "mthreads Wgrad must resolve to platform ownership"
        )
    if {
        "conv_wgrad_nd_kernel",
        "_conv_wgrad2d_p5_pack_image_kernel",
        "_conv_wgrad2d_stem_split_kernel",
        "_conv_wgrad2d_direct_split_kernel",
        "_conv_wgrad2d_im2row_kernel",
        "_conv_wgrad2d_rowmajor_kernel",
        "_conv_wgrad_nd_im2row_kernel",
        "_conv_wgrad_nd_rowmajor_kernel",
        "_conv_wgrad_nd_reduce_kernel",
        "_conv_wgrad2d_1x1_split_kernel",
        "_conv_wgrad2d_stem_reduce_kernel",
        "_conv_wgrad2d_p5_mm_kernel",
    }.difference(wgrad_candidate.functions):
        raise ValueError(
            "mthreads Wgrad candidate is missing required entry points"
        )
    wgrad_source = resolve_kernel_source(
        compiler_entry, wgrad_candidate
    ).resolve()
    if wgrad_source != platform_convolution_source:
        raise ValueError(
            "mthreads Dgrad and Wgrad resolve to different sources"
        )
    inputs["kernel:platform:convolution.py"] = platform_convolution_source
    add_square_candidate = select_kernel_candidate(
        "mthreads", "add_square"
    )
    if (
        add_square_candidate.ownership != "platform"
        or add_square_candidate.source != "composite.py"
        or "add_square_tensor_kernel"
        not in add_square_candidate.functions
    ):
        raise ValueError(
            "mthreads AddSquare candidate is not the platform fused kernel"
        )
    inputs["kernel:platform:composite.py"] = resolve_kernel_source(
        compiler_entry, add_square_candidate
    ).resolve()
    conv_bias_relu_candidate = select_kernel_candidate(
        "mthreads", "conv_bias_relu"
    )
    if (
        conv_bias_relu_candidate.ownership != "platform"
        or conv_bias_relu_candidate.source != "conv_bias_relu.py"
        or "conv_bias_relu_2d_kernel"
        not in conv_bias_relu_candidate.functions
    ):
        raise ValueError(
            "mthreads ConvBiasRelu candidate is not the platform fused kernel"
        )
    inputs["kernel:platform:conv_bias_relu.py"] = resolve_kernel_source(
        compiler_entry, conv_bias_relu_candidate
    ).resolve()
    tuning = provider_directory / "tuning/mthreads.yaml"
    if not tuning.is_file():
        raise ValueError("mthreads tuning policy is missing")
    inputs["tuning:mthreads.yaml"] = tuning.resolve()
    if environment_source is not None:
        inputs["environment:report"] = environment_source.resolve()
    install_metadata = _installed_metadata_path(environment_source)
    if install_metadata is not None:
        inputs["install:metadata"] = install_metadata

    resources = environment["resources"]
    modules = environment["python"]["modules"]
    for name in _identity_resource_names(environment_source):
        path, _, _ = _file_record(
            resources[name], f"resources.{name}"
        )
        inputs[f"environment:{name}"] = path
    executable, _, _ = _file_record(
        environment["python"]["executable"], "python.executable"
    )
    inputs["environment:python"] = executable
    for name in _PYTHON_MODULES:
        path, _, _ = _file_record(
            modules[name], f"python.modules.{name}"
        )
        inputs[f"environment:python:{name}"] = path
    for label, path in inputs.items():
        if not path.is_file():
            raise ValueError(f"compiler identity input is missing: {label}")
    return inputs


def _target_evidence(target_name: str) -> dict[str, int | str]:
    target = validate_target(target_name)
    match = TARGET_PATTERN.fullmatch(target)
    assert match is not None
    architecture = int(match.group(1))
    warp_size = int(match.group(2))
    return {
        "fingerprint": target,
        "architecture": architecture,
        "warp_size": warp_size,
    }


def compiler_identity_dependencies(
    target_name: str,
    execution_engine: str,
) -> tuple[Path, ...]:
    if execution_engine != SUPPORTED_ENGINE:
        raise ValueError("mthreads requires execution engine libtriton_jit")
    environment_source, environment = _validated_environment()
    _target_evidence(target_name)
    inputs = _identity_inputs(
        _compiler_entry_path(), environment_source, environment
    )
    return tuple(inputs[label] for label in sorted(inputs))


def build_compiler_identity(
    target_name: str,
    execution_engine: str,
) -> dict[str, Any]:
    if execution_engine != SUPPORTED_ENGINE:
        raise ValueError("mthreads requires execution engine libtriton_jit")
    compiler_entry = _compiler_entry_path()
    environment_source, environment = _validated_environment()
    inputs = _identity_inputs(
        compiler_entry, environment_source, environment
    )
    source_files = {
        label: _sha256_file(path)
        for label, path in sorted(inputs.items())
    }
    resources = environment["resources"]
    modules = environment["python"]["modules"]
    musa_root = resources.get("musa_root")
    if (
        not isinstance(musa_root, dict)
        or not isinstance(musa_root.get("realpath"), str)
    ):
        raise ValueError("mthreads MUSA root identity is invalid")
    root_name = Path(musa_root["realpath"]).name
    version_match = re.search(r"([0-9]+(?:\.[0-9]+)+)$", root_name)
    if version_match is None:
        raise ValueError("cannot determine MUSA toolkit version")
    jit_build_identity = _sha256_bytes(
        _canonical(
            {
                name: resources[name]["sha256"]
                for name in (
                    "triton_jit",
                    "triton_jit_config",
                    "triton_jit_gen_ssig",
                    "triton_jit_header",
                    "triton_jit_standalone_compile",
                )
            }
        )
    )
    payload: dict[str, Any] = {
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "execution_program_version": EXECUTION_PROGRAM_VERSION,
        "execution_engine": execution_engine,
        "target": _target_evidence(target_name),
        "environment_identity": environment["identity_sha256"],
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_modules": {
            name: {
                "version": modules[name]["version"],
                "sha256": modules[name]["sha256"],
            }
            for name in _PYTHON_MODULES
        },
        "musa_toolkit_version": version_match.group(1),
        "triton_jit_build_identity": jit_build_identity,
        "source_files": source_files,
    }
    payload["identity_sha256"] = _sha256_bytes(_canonical(payload))
    return payload
