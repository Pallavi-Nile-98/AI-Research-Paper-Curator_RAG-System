"""Guard against the mypy pre-commit hook drifting from pyproject.toml.

pre-commit builds an isolated environment per hook, so mypy there sees only the
packages listed in ``additional_dependencies`` -- not the project's real
dependencies. When a typed dependency is added to ``pyproject.toml`` and not to
the hook, mypy passes locally, where the virtualenv has the package, and fails
only at commit time with "Cannot find implementation or library stub". For
anything used as a decorator it cascades further, into "untyped decorator makes
function untyped" on every call site.

That drift happened four times while building this project, each time costing a
blocked commit and a confused minute. This test turns it into an ordinary test
failure naming the missing package.

Packages that mypy never needs to resolve are exempt: those given
``ignore_missing_imports`` in ``pyproject.toml`` are read from that config
rather than hard-coded, so the two cannot disagree either.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

# Dependencies the mypy hook genuinely does not need. Each exemption states
# why, because "it was failing" is not a reason and the next person adding a
# dependency has to be able to judge whether theirs belongs here.
EXEMPT_FROM_HOOK = frozenset(
    {
        # --- Never imported: named only inside SQLAlchemy connection strings
        # ("postgresql+asyncpg://"), so the driver is resolved at runtime and
        # mypy never sees it. ---------------------------------------------
        "asyncpg",
        "psycopg",
        # --- Imported, but its stubs come from a separate distribution.
        # types-defusedxml is in the hook; the runtime package is not needed
        # for type checking. ------------------------------------------------
        "defusedxml",
        # --- Optional [ml] extra, deliberately excluded: it pulls in PyTorch,
        # roughly 2 GB, to satisfy one lazily-imported symbol. Covered instead
        # by ignore_missing_imports in pyproject.toml. -----------------------
        "sentence-transformers",
        # --- Development tooling, not imported by the code being checked. ---
        "pytest-cov",
        "pytest-asyncio",
        "ruff",
        "mypy",
        "pre-commit",
        "pip-tools",
        "types-defusedxml",
    }
)


def _requirement_name(specifier: str) -> str:
    """Reduce a requirement string to its bare distribution name.

    ``opensearch-py[async]>=2.7`` becomes ``opensearch-py``; extras and version
    constraints are irrelevant to whether the package is present.
    """
    return re.split(r"[\[<>=!;~ ]", specifier.strip(), maxsplit=1)[0].lower()


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hook_dependencies() -> set[str]:
    """Distribution names listed under the mypy hook's additional_dependencies.

    Parsed with a regex rather than a YAML library, so this test needs no
    dependency of its own to verify a dependency list.
    """
    text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    block = re.search(
        r"additional_dependencies:\s*\n((?:\s*-\s*\S+\n)+)",
        text,
    )
    assert block is not None, "could not find additional_dependencies in the mypy hook"
    return {_requirement_name(line) for line in re.findall(r"-\s*(\S+)", block.group(1))}


@pytest.fixture(scope="module")
def ignored_modules(pyproject: dict[str, Any]) -> set[str]:
    """Top-level modules given ignore_missing_imports in pyproject.toml."""
    overrides = pyproject["tool"]["mypy"].get("overrides", [])
    ignored: set[str] = set()
    for override in overrides:
        if not override.get("ignore_missing_imports"):
            continue
        for pattern in override.get("module", []):
            ignored.add(pattern.split(".")[0].replace("_", "-").lower())
    return ignored


@pytest.mark.unit
class TestMypyHookDependencies:
    def test_every_runtime_dependency_is_available_to_the_hook(
        self, pyproject: dict[str, Any], hook_dependencies: set[str], ignored_modules: set[str]
    ) -> None:
        """A typed runtime dependency missing here fails only at commit time."""
        declared = {_requirement_name(spec) for spec in pyproject["project"]["dependencies"]}

        missing = sorted(
            name
            for name in declared
            if name not in hook_dependencies
            and name not in EXEMPT_FROM_HOOK
            # pytesseract is "pytesseract" as a distribution and as a module,
            # so the ignore list matches it directly.
            and name not in ignored_modules
        )

        assert not missing, (
            "These dependencies are in pyproject.toml but not in the mypy hook's "
            f"additional_dependencies: {missing}. Add them to "
            ".pre-commit-config.yaml, or exempt them here with a reason."
        )

    def test_the_hook_lists_nothing_unknown(
        self, pyproject: dict[str, Any], hook_dependencies: set[str]
    ) -> None:
        """Catches a stale entry left behind after a dependency is removed."""
        project = pyproject["project"]
        known = {_requirement_name(spec) for spec in project["dependencies"]}
        for extra in project.get("optional-dependencies", {}).values():
            known |= {_requirement_name(spec) for spec in extra}

        unknown = sorted(name for name in hook_dependencies if name not in known)
        assert not unknown, (
            f"The mypy hook lists packages that are not project dependencies: {unknown}"
        )


@pytest.mark.unit
class TestWarningPolicy:
    def test_warnings_are_errors_with_only_justified_exceptions(
        self, pyproject: dict[str, Any]
    ) -> None:
        """A blanket ignore would defeat the point of the strict policy.

        Turning warnings into errors is what surfaces a deprecation at the
        commit that introduces it rather than at some future upgrade. Each
        exception must be specific to one message.
        """
        filters = pyproject["tool"]["pytest"]["ini_options"]["filterwarnings"]
        assert filters[0] == "error", "the first filter must turn warnings into errors"

        for entry in filters[1:]:
            assert entry.startswith("ignore:"), f"unexpected filter form: {entry!r}"
            message = entry.removeprefix("ignore:").split(":")[0]
            assert message, f"blanket ignore with no message pattern: {entry!r}"
