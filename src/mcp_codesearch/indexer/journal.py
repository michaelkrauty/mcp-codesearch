"""Durable write-ahead intents for cross-store indexing operations."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vector_core import file_lock

from mcp_codesearch.settings import settings

IntentOperation = Literal["index", "delete"]


@dataclass(frozen=True)
class IndexIntent:
    """An operation that may have left Qdrant and the vocabulary out of sync."""

    collection: str
    codebase_path: str
    operation: IntentOperation
    pending_paths: tuple[str, ...]
    started_at: float


class IndexIntentStore:
    """Small SQLite journal shared by every codesearch server process.

    An intent is committed before either Qdrant or the global vocabulary is
    mutated. It remains until both stores are known to agree. SQLite connections
    are deliberately short-lived so a process crash cannot retain a lock.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        namespace: str = "default",
    ) -> None:
        if not namespace:
            raise ValueError("index journal namespace must be non-empty")
        self.namespace = namespace
        self.db_path = db_path or (settings.cache_dir / "codesearch_index_journal_v2.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.db_path, timeout=30.0):
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_intents (
                    namespace TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    codebase_path TEXT NOT NULL,
                    operation TEXT NOT NULL CHECK (operation IN ('index', 'delete')),
                    pending_paths TEXT NOT NULL DEFAULT '[]',
                    started_at REAL NOT NULL,
                    PRIMARY KEY (namespace, collection)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_metadata (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO journal_metadata (namespace, key, value) "
                "VALUES (?, 'generation', 0)",
                (self.namespace,),
            )
            conn.commit()

    def mark(
        self,
        collection: str,
        codebase_path: str,
        operation: IntentOperation,
    ) -> None:
        """Begin or replace an operation, clearing any prior batch marker."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO index_intents
                    (namespace, collection, codebase_path, operation,
                     pending_paths, started_at)
                VALUES (?, ?, ?, ?, '[]', ?)
                ON CONFLICT(namespace, collection) DO UPDATE SET
                    codebase_path = excluded.codebase_path,
                    operation = excluded.operation,
                    pending_paths = '[]',
                    started_at = excluded.started_at
                """,
                (self.namespace, collection, codebase_path, operation, time.time()),
            )
            conn.commit()

    def set_pending_paths(self, collection: str, paths: list[str]) -> None:
        """Record paths immediately before a potentially partial Qdrant write."""
        encoded = json.dumps(sorted(set(paths)), separators=(",", ":"))
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE index_intents SET pending_paths = ? WHERE namespace = ? AND collection = ?",
                (encoded, self.namespace, collection),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise RuntimeError(f"No active indexing intent for {collection!r}")
            conn.commit()

    def clear_pending_paths(self, collection: str) -> None:
        """Mark the current Qdrant batch as completely applied."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE index_intents SET pending_paths = '[]' "
                "WHERE namespace = ? AND collection = ?",
                (self.namespace, collection),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise RuntimeError(f"No active indexing intent for {collection!r}")
            conn.commit()

    def get(self, collection: str) -> IndexIntent | None:
        """Return one pending intent."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT collection, codebase_path, operation, pending_paths, started_at
                FROM index_intents WHERE namespace = ? AND collection = ?
                """,
                (self.namespace, collection),
            ).fetchone()
        return self._from_row(row) if row else None

    def list(self) -> list[IndexIntent]:
        """Return all pending intents in deterministic order."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT collection, codebase_path, operation, pending_paths, started_at
                FROM index_intents WHERE namespace = ? ORDER BY collection
                """,
                (self.namespace,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def clear(self, collection: str) -> None:
        """Commit completion and advance the global search-cache generation."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM index_intents WHERE namespace = ? AND collection = ?",
                (self.namespace, collection),
            )
            if cursor.rowcount:
                self._bump_generation(conn, self.namespace)
            conn.commit()

    def generation(self) -> int:
        """Return the generation shared by every process-local result cache."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT value FROM journal_metadata WHERE namespace = ? AND key = 'generation'",
                (self.namespace,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Index journal has no generation record")
        return int(row[0])

    def bump_generation(self) -> int:
        """Advance the generation after a direct, non-intent reconciliation."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            value = self._bump_generation(conn, self.namespace)
            conn.commit()
        return value

    @staticmethod
    def _bump_generation(conn: sqlite3.Connection, namespace: str) -> int:
        conn.execute(
            "UPDATE journal_metadata SET value = value + 1 "
            "WHERE namespace = ? AND key = 'generation'",
            (namespace,),
        )
        row = conn.execute(
            "SELECT value FROM journal_metadata WHERE namespace = ? AND key = 'generation'",
            (namespace,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Index journal has no generation record")
        return int(row[0])

    @staticmethod
    def _from_row(row: tuple[str, str, str, str, float]) -> IndexIntent:
        raw_paths = json.loads(row[3])
        if not isinstance(raw_paths, list) or not all(isinstance(path, str) for path in raw_paths):
            raise RuntimeError(f"Invalid pending path list for {row[0]!r}")
        operation: IntentOperation
        if row[2] == "index":
            operation = "index"
        elif row[2] == "delete":
            operation = "delete"
        else:
            raise RuntimeError(f"Invalid intent operation {row[2]!r}")
        return IndexIntent(
            collection=row[0],
            codebase_path=row[1],
            operation=operation,
            pending_paths=tuple(raw_paths),
            started_at=row[4],
        )
