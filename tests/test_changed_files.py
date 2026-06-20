"""Unit tests for _changed_files_since (git diff resolution; no Qdrant)."""
import subprocess
from pathlib import Path

import pytest

from mcp_codesearch.tools.search import _changed_files_since


def _git(repo: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def diverged_repo(tmp_path: Path) -> str:
    """A repo where `feature` (current branch) and `main` diverged: feature
    added b.py, main independently added c.py afterwards."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t.com")
    _git(str(repo), "config", "user.name", "T")
    (repo / "base.py").write_text("x = 1\n")
    _git(str(repo), "add", ".")
    _git(str(repo), "commit", "-qm", "initial")
    _git(str(repo), "branch", "-M", "main")
    _git(str(repo), "checkout", "-q", "-b", "feature")
    (repo / "b.py").write_text("b = 1\n")
    _git(str(repo), "add", ".")
    _git(str(repo), "commit", "-qm", "feature change")
    _git(str(repo), "checkout", "-q", "main")
    (repo / "c.py").write_text("c = 1\n")
    _git(str(repo), "add", ".")
    _git(str(repo), "commit", "-qm", "main change")
    _git(str(repo), "checkout", "-q", "feature")
    return str(repo)


def test_since_branch_uses_merge_base_not_tip(diverged_repo: str) -> None:
    """since='main' returns what the user changed since diverging from main,
    NOT files main advanced on after divergence."""
    changed, error = _changed_files_since(diverged_repo, diverged_repo, "main")
    assert error == ""
    assert "b.py" in changed       # the user's change
    assert "c.py" not in changed   # advanced on main after divergence, never by the user


def test_since_branch_includes_uncommitted(diverged_repo: str) -> None:
    """Uncommitted working-tree changes since the merge-base are included."""
    (Path(diverged_repo) / "base.py").write_text("x = 2\n")
    changed, error = _changed_files_since(diverged_repo, diverged_repo, "main")
    assert error == ""
    assert "base.py" in changed
    assert "c.py" not in changed


def test_since_ancestor_revision_unchanged(diverged_repo: str) -> None:
    """For an ancestor revision (HEAD~1), merge-base == the revision, so the
    result is the commits since it (the prior behavior for ancestors)."""
    changed, error = _changed_files_since(diverged_repo, diverged_repo, "HEAD~1")
    assert error == ""
    assert "b.py" in changed


def test_invalid_revision_returns_error(diverged_repo: str) -> None:
    changed, error = _changed_files_since(diverged_repo, diverged_repo, "no_such_ref")
    assert changed == set()
    assert error.startswith("Error:")
