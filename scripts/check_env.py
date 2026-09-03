"""Report whether this machine has everything needed to run the project.

Run this first after cloning. It replaces the "why doesn't it work on my machine"
round-trip with a single command that names exactly what is missing.

    python scripts/check_env.py

Exit code is 0 when every *required* tool is present, 1 otherwise. Optional tools
produce a warning rather than a failure, because several of them (Ollama, Tesseract,
Poppler) run inside containers and are not needed on the host at all.
"""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "warn", "missing"]

_SYMBOLS: dict[Status, str] = {"ok": "[ OK ]", "warn": "[WARN]", "missing": "[MISS]"}

# Minimum Python version, kept in step with `requires-python` in pyproject.toml.
_MIN_PYTHON = (3, 12)

# Ports the local stack expects to be free on the host.
_PORTS: tuple[tuple[int, str], ...] = (
    (5433, "Postgres (container -> host)"),
    (9200, "OpenSearch REST"),
    (9600, "OpenSearch performance analyzer"),
    (5601, "OpenSearch Dashboards"),
    (11434, "Ollama"),
    (8002, "FastAPI"),
    (5173, "Vite dev server"),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    status: Status
    detail: str


def _run(executable: str, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a resolved executable, returning None if it is absent or cannot be launched."""
    resolved = shutil.which(executable)
    if resolved is None:
        return None
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, absolute path from shutil.which
            [resolved, *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _first_line(text: str) -> str:
    """Return the first non-blank line of ``text``, or an empty string."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _version_of(executable: str, args: list[str]) -> str | None:
    """Return the first line of ``executable``'s version output, or None if unusable.

    Exit codes are not trusted here: some tools report failures on stderr while still
    exiting 0. Blank output is therefore treated as failure, and stderr is consulted
    only when stdout has nothing meaningful in it.
    """
    completed = _run(executable, args)
    if completed is None:
        return None
    return _first_line(completed.stdout) or _first_line(completed.stderr) or None


def check_python() -> CheckResult:
    """Verify the running interpreter satisfies the project's minimum version."""
    current = sys.version_info[:2]
    detail = f"{platform.python_version()} ({sys.executable})"
    if current < _MIN_PYTHON:
        needed = ".".join(str(part) for part in _MIN_PYTHON)
        return CheckResult("Python", "missing", f"{detail} - need >= {needed}")
    return CheckResult("Python", "ok", detail)


def check_tool(
    name: str,
    executable: str,
    args: list[str],
    *,
    required: bool,
    note: str = "",
) -> CheckResult:
    """Check for a command-line tool and capture its version."""
    version = _version_of(executable, args)
    if version is not None:
        return CheckResult(name, "ok", version)
    detail = f"not found on PATH{f' - {note}' if note else ''}"
    return CheckResult(name, "missing" if required else "warn", detail)


def check_docker_daemon() -> CheckResult:
    """Check that the Docker daemon is reachable, not merely that the CLI exists.

    ``docker info`` exits 0 on Windows even when Docker Desktop is stopped, printing a
    blank line to stdout and the real error to stderr. The only trustworthy signal is a
    non-blank server version on *stdout*, so that is what is required here.
    """
    if shutil.which("docker") is None:
        return CheckResult("Docker daemon", "missing", "docker CLI not on PATH")

    completed = _run("docker", ["info", "--format", "{{.ServerVersion}}"])
    server_version = _first_line(completed.stdout) if completed is not None else ""
    if not server_version:
        return CheckResult("Docker daemon", "missing", "not reachable - start Docker Desktop")
    return CheckResult("Docker daemon", "ok", f"server {server_version}")


def check_port(port: int, label: str) -> CheckResult:
    """Report whether a TCP port on localhost is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        in_use = probe.connect_ex(("127.0.0.1", port)) == 0
    if in_use:
        return CheckResult(f"port {port}", "warn", f"{label} - already in use")
    return CheckResult(f"port {port}", "ok", f"{label} - free")


def check_env_file(repo_root: Path) -> CheckResult:
    """Report whether a local .env exists; absence is fine because defaults apply."""
    if (repo_root / ".env").exists():
        return CheckResult(".env", "ok", "present")
    return CheckResult(".env", "warn", "absent - copy .env.example to .env")


def collect_results(repo_root: Path) -> list[CheckResult]:
    """Run every check and return the results in display order."""
    return [
        check_python(),
        check_tool("Git", "git", ["--version"], required=True),
        check_tool("Docker CLI", "docker", ["--version"], required=True),
        check_docker_daemon(),
        check_tool("Docker Compose", "docker-compose", ["version"], required=True),
        check_tool(
            "Node.js",
            "node",
            ["--version"],
            required=False,
            note="needed from Phase 3 for the React client",
        ),
        check_tool("npm", "npm", ["--version"], required=False, note="needed from Phase 3"),
        check_tool(
            "Ollama",
            "ollama",
            ["--version"],
            required=False,
            note="runs in Docker; a host install is optional",
        ),
        check_tool(
            "Tesseract",
            "tesseract",
            ["--version"],
            required=False,
            note="OCR runs in the ingestion container",
        ),
        check_tool(
            "Poppler",
            "pdftoppm",
            ["-v"],
            required=False,
            note="OCR runs in the ingestion container",
        ),
        check_env_file(repo_root),
        *(check_port(port, label) for port, label in _PORTS),
    ]


def main() -> int:
    """Print the environment report and return a process exit code."""
    repo_root = Path(__file__).resolve().parent.parent
    results = collect_results(repo_root)
    width = max(len(result.name) for result in results)

    print(f"Environment check for {repo_root}")
    print(f"Platform: {platform.platform()}")
    print("-" * 78)
    for result in results:
        print(f"{_SYMBOLS[result.status]} {result.name.ljust(width)}  {result.detail}")
    print("-" * 78)

    missing = [result.name for result in results if result.status == "missing"]
    warnings = [result.name for result in results if result.status == "warn"]

    if missing:
        print(f"MISSING (required): {', '.join(missing)}")
    if warnings:
        print(f"Warnings (optional or informational): {', '.join(warnings)}")
    if not missing:
        print("All required tools are present.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
