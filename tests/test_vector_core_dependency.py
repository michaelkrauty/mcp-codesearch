from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vector_core_dependency_pinned_to_v1_0_3() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    vector_specs = [dep for dep in dependencies if dep.startswith("vector-core @ git+")]

    assert vector_specs == [
        "vector-core @ git+https://github.com/michaelkrauty/vector-core.git@v1.0.3"
    ]

    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    assert uv_sources.get("vector-core", {}).get("tag") == "v1.0.3"


def test_readme_install_example_uses_current_vector_core_pin() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "vector-core.git@v1.0.3" in readme
    assert "mcp-codesearch.git@v1.0.0" not in readme

def test_project_version_is_v1_0_1() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.0.1"
