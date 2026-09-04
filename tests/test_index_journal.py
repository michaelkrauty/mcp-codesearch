"""Tests for the durable cross-store indexing journal."""

from __future__ import annotations

from multiprocessing import Process

import pytest

from mcp_codesearch.indexer.journal import IndexIntentStore


def _mark_from_process(db_path, collection: str) -> None:
    store = IndexIntentStore(db_path)
    store.mark(collection, f"/{collection}", "index")


def test_intent_round_trip_and_batch_state(tmp_path) -> None:
    store = IndexIntentStore(tmp_path / "journal.db")
    assert store.generation() == 0

    store.mark("codesearch_abc", "/repo", "index")
    store.set_pending_paths("codesearch_abc", ["b.py", "a.py", "a.py"])

    intent = store.get("codesearch_abc")
    assert intent is not None
    assert intent.codebase_path == "/repo"
    assert intent.operation == "index"
    assert intent.pending_paths == ("a.py", "b.py")

    store.clear_pending_paths("codesearch_abc")
    assert store.get("codesearch_abc").pending_paths == ()  # type: ignore[union-attr]

    store.clear("codesearch_abc")
    assert store.get("codesearch_abc") is None
    assert store.generation() == 1
    store.clear("codesearch_abc")
    assert store.generation() == 1
    assert store.bump_generation() == 2


def test_mark_replaces_prior_operation_and_paths(tmp_path) -> None:
    store = IndexIntentStore(tmp_path / "journal.db")
    store.mark("codesearch_abc", "/old", "index")
    store.set_pending_paths("codesearch_abc", ["partial.py"])

    store.mark("codesearch_abc", "/new", "delete")

    intent = store.get("codesearch_abc")
    assert intent is not None
    assert intent.codebase_path == "/new"
    assert intent.operation == "delete"
    assert intent.pending_paths == ()


def test_batch_state_requires_an_active_intent(tmp_path) -> None:
    store = IndexIntentStore(tmp_path / "journal.db")

    with pytest.raises(RuntimeError, match="No active indexing intent"):
        store.set_pending_paths("codesearch_missing", ["file.py"])


def test_separate_processes_share_the_journal(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    IndexIntentStore(db_path)
    processes = [
        Process(target=_mark_from_process, args=(db_path, f"codesearch_{index}"))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    intents = IndexIntentStore(db_path).list()
    assert [intent.collection for intent in intents] == [
        "codesearch_0",
        "codesearch_1",
        "codesearch_2",
        "codesearch_3",
    ]


def test_namespaces_isolate_intents_and_generations(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    first = IndexIntentStore(db_path, namespace="deployment-a")
    second = IndexIntentStore(db_path, namespace="deployment-b")

    first.mark("codesearch_same", "/a", "index")
    assert second.get("codesearch_same") is None
    first.clear("codesearch_same")

    assert first.generation() == 1
    assert second.generation() == 0
