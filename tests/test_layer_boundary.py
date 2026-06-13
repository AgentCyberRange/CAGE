"""Layer-2 -> Layer-1 import boundary.

Benchmarks (examples/) may only import the public framework surface. This is
the machine-enforced version of the CLAUDE.md rule "must not reach into
framework internals beyond the ABCs":

* only modules in ``PUBLIC_MODULES`` may be imported from ``cage``;
* no underscore-private names may be imported from any ``cage`` module;
* ``if TYPE_CHECKING:`` imports are exempt — type-only imports create no
  runtime coupling.

``import-linter`` cannot see examples/ (its root package is ``cage``), so this
test is the only fence on that edge. If a benchmark legitimately needs a new
framework capability, the fix is to promote a public API (export it from the
owning package's ``__init__``) — never to widen this allowlist with a deep
internal module.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: The public framework surface for benchmark authors. Package roots only —
#: each re-exports its public names from its ``__init__``.
PUBLIC_MODULES = {
    "cage.artifacts",
    "cage.benchmarks",
    "cage.config",
    "cage.contracts",
    "cage.models",
    "cage.scoring",
    "cage.target",
    "cage.target.adapters",
}


def _example_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "examples"
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if "datasets" not in path.parts
    ]


def _cage_imports(tree: ast.Module) -> list[tuple[int, str, list[str], bool]]:
    """Yield ``(lineno, module, imported_names, type_checking_only)``."""

    found: list[tuple[int, str, list[str], bool]] = []

    def visit(node: ast.AST, type_checking: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_type_checking = type_checking
            if isinstance(child, ast.If):
                test = child.test
                is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                )
                if is_tc:
                    for body_node in child.body:
                        visit_stmt(body_node, True)
                    for else_node in child.orelse:
                        visit_stmt(else_node, type_checking)
                    continue
            visit_stmt(child, child_type_checking)

    def visit_stmt(node: ast.AST, type_checking: bool) -> None:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("cage"):
            found.append(
                (node.lineno, node.module, [alias.name for alias in node.names], type_checking)
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cage"):
                    found.append((node.lineno, alias.name, [], type_checking))
        visit(node, type_checking)

    visit(tree, False)
    return found


def test_examples_import_only_the_public_framework_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in _example_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(repo_root)
        for lineno, module, names, type_checking in _cage_imports(tree):
            private = [name for name in names if name.startswith("_")]
            if private:
                offenders.append(f"{rel}:{lineno}: private name(s) {private} from {module}")
                continue
            if type_checking:
                continue
            if module not in PUBLIC_MODULES:
                offenders.append(f"{rel}:{lineno}: non-public module {module}")
    assert offenders == [], "\n".join(offenders)
