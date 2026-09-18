"""MThreads codegen resources implementation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from flagdnn_codegen.kernel_registry import materialize_kernel_source


def _bound_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_bound_names(element))
        return names
    return set()


def _definition_names(statement: ast.stmt) -> set[str]:
    if isinstance(
        statement,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
    ):
        return {statement.name}
    if isinstance(statement, ast.Assign):
        names: set[str] = set()
        for target in statement.targets:
            names.update(_bound_names(target))
        return names
    if isinstance(statement, ast.AnnAssign):
        return _bound_names(statement.target)
    return set()


def _statement_start(statement: ast.stmt) -> int:
    decorators = getattr(statement, "decorator_list", ())
    return min(
        [statement.lineno] + [decorator.lineno for decorator in decorators]
    )


def _materialize_mthreads_kernel_source(
    source_path: Path, candidate: Any
) -> bytes:
    """Materialize a compact MThreads artifact for oversized common modules.

    The common registry intentionally exposes only production entry points,
    while convolution.py also contains many auxiliary policy experiments.  A
    standalone MThreads artifact keeps the registry entries and the transitive
    closure of their top-level Python dependencies.  Other source families use
    the canonical module unchanged.
    """

    source_bytes = materialize_kernel_source(source_path, candidate)
    if candidate.source != "convolution.py":
        return source_bytes

    try:
        source_text = source_bytes.decode("utf-8")
        module = ast.parse(source_text, filename=str(source_path))
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValueError(
            "common convolution source cannot be candidate-sliced"
        ) from error

    definitions: dict[str, ast.stmt] = {}
    for statement in module.body:
        for name in _definition_names(statement):
            if name in definitions:
                raise ValueError(
                    f"common convolution source redefines {name!r}"
                )
            definitions[name] = statement

    selected = set(candidate.functions)
    missing = selected.difference(definitions)
    if missing:
        raise ValueError(
            "common convolution source is missing registry entry points: "
            + ", ".join(sorted(missing))
        )

    pending = list(selected)
    while pending:
        name = pending.pop()
        for item in ast.walk(definitions[name]):
            if (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id in definitions
                and item.id not in selected
            ):
                selected.add(item.id)
                pending.append(item.id)

    selected_statements = {
        id(statement)
        for name, statement in definitions.items()
        if name in selected
    }
    retained: list[ast.stmt] = []
    for index, statement in enumerate(module.body):
        module_docstring = (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        if (
            module_docstring
            or isinstance(statement, (ast.Import, ast.ImportFrom))
            or id(statement) in selected_statements
        ):
            retained.append(statement)
    if not retained:
        raise ValueError("common convolution source slicing retained nothing")

    lines = source_text.splitlines(keepends=True)
    first_statement = min(_statement_start(item) for item in module.body)
    segments = ["".join(lines[: first_statement - 1]).rstrip()]
    for statement in retained:
        if statement.end_lineno is None:
            raise ValueError(
                "common convolution source lacks AST end positions"
            )
        start = _statement_start(statement)
        segments.append(
            "".join(lines[start - 1 : statement.end_lineno]).rstrip()
        )
    materialized = (
        "\n\n".join(segment for segment in segments if segment) + "\n"
    )
    materialized_bytes = materialized.encode("utf-8")

    materialized_module = ast.parse(
        materialized_bytes, filename=str(source_path)
    )
    materialized_functions = {
        item.name
        for item in materialized_module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not set(candidate.functions).issubset(materialized_functions):
        raise ValueError(
            "candidate-sliced convolution source lost an entry point"
        )
    return materialized_bytes
