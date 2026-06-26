from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vector_core_dependency_is_pinned_to_v1_2_11() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    vector_specs = [dep for dep in dependencies if dep.startswith("vector-core @ git+")]

    assert vector_specs == [
        "vector-core @ git+https://github.com/michaelkrauty/vector-core.git@v1.2.11"
    ]

    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    assert uv_sources.get("vector-core", {}).get("tag") == "v1.2.11"


def test_readme_install_example_uses_current_vector_core_pin() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "vector-core.git@v1.2.11" in readme
    assert "mcp-codesearch.git@v1.0.0" not in readme

def test_project_version_is_v1_6_19() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.6.19"


def test_runtime_version_matches_project_metadata() -> None:
    import mcp_codesearch

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert mcp_codesearch.__version__ == pyproject["project"]["version"]
