from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.oracles import build


def _make_layout(root: Path, prefix: str = "") -> build.CannLayout:
    base = root / prefix if prefix else root
    include_dir = base / "include"
    lib_dir = base / "lib64"
    (include_dir / "aclnnop").mkdir(parents=True)
    lib_dir.mkdir(parents=True)
    for name in ("aclnn_add.h", "aclnn_abs.h"):
        (include_dir / "aclnnop" / name).write_text(
            "// fake\n", encoding="utf-8"
        )
    (lib_dir / "libascendcl.so").touch()
    (lib_dir / "libopapi.so").touch()
    return build.CannLayout(root, include_dir, lib_dir)


def _install_successful_fake_compiler(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(
                returncode=0, stdout="fake-cxx 1.0\n", stderr=""
            )
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"test-so")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/c++")
    monkeypatch.setattr(build.subprocess, "run", fake_run)
    return commands


def test_find_cann_layout_prefers_environment_root(tmp_path):
    expected = _make_layout(tmp_path / "cann", "aarch64-linux")

    actual = build.find_cann_layout(
        environ={"ASCEND_HOME_PATH": str(expected.root)},
        machine="aarch64",
    )

    assert actual == expected


def test_find_cann_layout_rejects_missing_abs_header(tmp_path, monkeypatch):
    layout = _make_layout(tmp_path / "cann")
    (layout.include_dir / "aclnnop" / "aclnn_abs.h").unlink()
    monkeypatch.setattr(build, "_DEFAULT_ROOTS", ())

    with pytest.raises(RuntimeError, match="Could not find CANN ACLNN"):
        build.find_cann_layout(
            environ={"ASCEND_HOME_PATH": str(layout.root)},
            machine="aarch64",
        )


def test_build_compiles_sorted_sources_once(tmp_path, monkeypatch):
    layout = _make_layout(tmp_path / "cann")
    native = tmp_path / "native"
    source_b = native / "ops" / "z.cpp"
    source_a = native / "common" / "a.cpp"
    header = native / "common" / "oracle_common.h"
    for path, text in (
        (source_b, "int z = 1;\n"),
        (source_a, "int a = 1;\n"),
        (header, "#pragma once\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    commands = _install_successful_fake_compiler(monkeypatch)
    output = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source_b, source_a, source_b),
        header_paths=(header,),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )

    compile_command = commands[-1]
    assert compile_command.count(str(source_a.resolve())) == 1
    assert compile_command.count(str(source_b.resolve())) == 1
    assert compile_command.index(
        str(source_a.resolve())
    ) < compile_command.index(str(source_b.resolve()))
    assert output.name == "libflagdnn_test_aclnn.so"


def test_default_native_discovery_includes_all_operator_sources_and_headers():
    sources = tuple(
        path.relative_to(build._NATIVE_ROOT).as_posix()
        for path in build._default_source_paths()
    )
    headers = tuple(
        path.relative_to(build._NATIVE_ROOT).as_posix()
        for path in build._default_header_paths()
    )

    assert sources == (
        "common/oracle_common.cpp",
        "ops/abs.cpp",
        "ops/add.cpp",
    )
    assert headers == ("common/oracle_common.h",)


def test_common_header_change_invalidates_cache(tmp_path, monkeypatch):
    layout = _make_layout(tmp_path / "cann")
    source = tmp_path / "native" / "ops" / "add.cpp"
    header = tmp_path / "native" / "common" / "oracle_common.h"
    source.parent.mkdir(parents=True)
    header.parent.mkdir(parents=True)
    source.write_text("int add = 1;\n", encoding="utf-8")
    header.write_text("#define VERSION 1\n", encoding="utf-8")
    _install_successful_fake_compiler(monkeypatch)

    first = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source,),
        header_paths=(header,),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )
    header.write_text("#define VERSION 2\n", encoding="utf-8")
    second = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source,),
        header_paths=(header,),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )

    assert first.parent != second.parent


def test_build_uses_content_addressed_cache(tmp_path, monkeypatch):
    layout = _make_layout(tmp_path / "cann")
    source = tmp_path / "aclnn_oracle.cpp"
    source.write_text("int test_source = 1;\n", encoding="utf-8")
    compile_calls = []

    def fake_run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(
                returncode=0, stdout="fake-cxx 1.0\n", stderr=""
            )
        compile_calls.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"test-so")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/c++")
    monkeypatch.setattr(build.subprocess, "run", fake_run)

    first = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source,),
        header_paths=(),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )
    second = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source,),
        header_paths=(),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )

    assert first == second
    assert first.read_bytes() == b"test-so"
    assert len(compile_calls) == 1


def test_build_uses_unique_temporary_paths_for_same_key(tmp_path, monkeypatch):
    layout = _make_layout(tmp_path / "cann")
    source = tmp_path / "aclnn_oracle.cpp"
    source.write_text("int test_source = 1;\n", encoding="utf-8")
    temporary_outputs = []
    real_replace = build.os.replace

    def fake_run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(
                returncode=0, stdout="fake-cxx 1.0\n", stderr=""
            )
        temporary = Path(command[command.index("-o") + 1])
        temporary_outputs.append(temporary)
        temporary.write_bytes(b"test-so")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def publish_without_retaining_cache(temporary, output):
        real_replace(temporary, output)
        Path(output).unlink()

    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/c++")
    monkeypatch.setattr(build.subprocess, "run", fake_run)
    monkeypatch.setattr(build.os, "replace", publish_without_retaining_cache)

    first = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source,),
        header_paths=(),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )
    second = build.build_aclnn_oracle(
        layout=layout,
        source_paths=(source,),
        header_paths=(),
        build_root=tmp_path / "build",
        environ={"CXX": "c++"},
    )

    assert first == second
    assert len(temporary_outputs) == 2
    assert temporary_outputs[0] != temporary_outputs[1]
    assert all(path.parent == first.parent for path in temporary_outputs)
    assert all(not path.exists() for path in temporary_outputs)


def test_compiler_not_found_reports_full_build_diagnostics(
    tmp_path, monkeypatch
):
    layout = _make_layout(tmp_path / "cann")
    source = tmp_path / "aclnn_oracle.cpp"
    source.write_text("int test_source = 1;\n", encoding="utf-8")

    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build.platform, "machine", lambda: "test-arch")

    with pytest.raises(RuntimeError) as exc_info:
        build.build_aclnn_oracle(
            layout=layout,
            source_paths=(source,),
            header_paths=(),
            build_root=tmp_path / "build",
            environ={"CXX": "missing-c++ --test-flag"},
        )

    message = str(exc_info.value)
    assert "missing-c++ --test-flag" in message
    assert "compiler:" in message
    assert str(layout.include_dir) in message
    assert str(layout.lib_dir) in message
    assert "test-arch" in message
    assert "stdout:\nunavailable" in message
    assert "stderr:\nunavailable" in message


def test_compiler_version_failure_reports_full_build_diagnostics(
    tmp_path, monkeypatch
):
    layout = _make_layout(tmp_path / "cann")
    source = tmp_path / "aclnn_oracle.cpp"
    source.write_text("int test_source = 1;\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command == ["/usr/bin/c++", "--version"]
        return SimpleNamespace(
            returncode=1,
            stdout="version stdout",
            stderr="version stderr",
        )

    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/c++")
    monkeypatch.setattr(build.subprocess, "run", fake_run)
    monkeypatch.setattr(build.platform, "machine", lambda: "test-arch")

    with pytest.raises(RuntimeError) as exc_info:
        build.build_aclnn_oracle(
            layout=layout,
            source_paths=(source,),
            header_paths=(),
            build_root=tmp_path / "build",
            environ={"CXX": "c++"},
        )

    message = str(exc_info.value)
    assert "/usr/bin/c++ --version" in message
    assert "compiler: /usr/bin/c++" in message
    assert str(layout.include_dir) in message
    assert str(layout.lib_dir) in message
    assert "test-arch" in message
    assert "version stdout" in message
    assert "version stderr" in message


def test_build_failure_reports_command_and_cann_paths(tmp_path, monkeypatch):
    layout = _make_layout(tmp_path / "cann")
    source = tmp_path / "aclnn_oracle.cpp"
    source.write_text("broken source\n", encoding="utf-8")
    temporary_outputs = []

    def fake_run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(
                returncode=0, stdout="fake-cxx 1.0\n", stderr=""
            )
        temporary = Path(command[command.index("-o") + 1])
        temporary_outputs.append(temporary)
        temporary.write_bytes(b"partial output")
        return SimpleNamespace(
            returncode=1,
            stdout="compiler stdout",
            stderr="compiler stderr",
        )

    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/c++")
    monkeypatch.setattr(build.subprocess, "run", fake_run)
    monkeypatch.setattr(build.platform, "machine", lambda: "test-arch")

    with pytest.raises(RuntimeError) as exc_info:
        build.build_aclnn_oracle(
            layout=layout,
            source_paths=(source,),
            header_paths=(),
            build_root=tmp_path / "build",
            environ={"CXX": "c++"},
        )

    message = str(exc_info.value)
    assert "/usr/bin/c++" in message
    assert str(layout.include_dir) in message
    assert str(layout.lib_dir) in message
    assert "test-arch" in message
    assert "compiler stdout" in message
    assert "compiler stderr" in message
    assert len(temporary_outputs) == 1
    assert not temporary_outputs[0].exists()
