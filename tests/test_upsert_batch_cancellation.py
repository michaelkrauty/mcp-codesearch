"""A failed batch upsert must not leave sibling writes running.

`asyncio.gather` re-raises the first failure without stopping the others. The
caller recovers by deleting ambiguous paths and rebuilding vocabulary from the
remaining points, so every remotely accepted write must finish before recovery
can begin. Cancellation waits for the Qdrant request rather than abandoning it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_codesearch.storage import qdrant as qdrant_module
from mcp_codesearch.storage.qdrant import QdrantStorage


class TestUpsertBatchQuiescesSiblings:
    async def test_a_failing_batch_waits_for_the_others(self, monkeypatch):
        # Keep the batch timeout short: if cancellation regresses, the test
        # should fail quickly rather than block on the production default.
        monkeypatch.setattr(qdrant_module.settings, "upsert_batch_timeout", 10.0)

        storage = QdrantStorage()

        first_started = asyncio.Event()
        slow_completed = False
        call_index = 0

        async def upsert(collection, batch, **_kwargs):
            nonlocal slow_completed, call_index
            index = call_index
            call_index += 1
            if index == 0:
                # The long-running sibling, still in flight when the other
                # batch fails.
                first_started.set()
                await asyncio.sleep(0.3)
                slow_completed = True
            else:
                await first_started.wait()
                raise RuntimeError("upsert rejected")

        client = MagicMock()
        client.upsert = AsyncMock(side_effect=upsert)
        storage._get_client = AsyncMock(return_value=client)

        # batch_size=1 puts each point in its own sub-batch; max_retries=1
        # keeps the failing one from retrying past the assertion.
        points = [MagicMock(), MagicMock()]

        with pytest.raises(RuntimeError, match="upsert rejected"):
            await storage.upsert_batch("col", points, batch_size=1, concurrency=2, max_retries=1)

        # The error is not returned until the sibling request is known to have
        # completed, so recovery cannot race it and delete too early.
        assert slow_completed is True

    async def test_caller_cancellation_waits_for_accepted_write(self, monkeypatch):
        monkeypatch.setattr(qdrant_module.settings, "upsert_batch_timeout", 10.0)
        storage = QdrantStorage()
        started = asyncio.Event()
        release = asyncio.Event()
        completed = False

        async def upsert(collection, batch, **_kwargs):
            nonlocal completed
            _ = collection, batch
            started.set()
            await release.wait()
            completed = True

        client = MagicMock()
        client.upsert = AsyncMock(side_effect=upsert)
        storage._get_client = AsyncMock(return_value=client)
        task = asyncio.create_task(storage.upsert_batch("col", [MagicMock()], max_retries=1))
        await started.wait()

        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed is True

    async def test_repeated_cancellation_cannot_interrupt_sibling_quiescence(self, monkeypatch):
        monkeypatch.setattr(qdrant_module.settings, "upsert_batch_timeout", 10.0)
        storage = QdrantStorage()
        slow_started = asyncio.Event()
        failure_returned = asyncio.Event()
        release = asyncio.Event()
        slow_completed = False
        call_index = 0

        async def upsert(collection, batch, **_kwargs):
            nonlocal call_index, slow_completed
            _ = collection, batch
            index = call_index
            call_index += 1
            if index == 0:
                slow_started.set()
                await release.wait()
                slow_completed = True
                return
            await slow_started.wait()
            failure_returned.set()
            raise RuntimeError("upsert rejected")

        client = MagicMock()
        client.upsert = AsyncMock(side_effect=upsert)
        storage._get_client = AsyncMock(return_value=client)
        task = asyncio.create_task(
            storage.upsert_batch(
                "col",
                [MagicMock(), MagicMock()],
                batch_size=1,
                concurrency=2,
                max_retries=1,
            )
        )
        await failure_returned.wait()
        await asyncio.sleep(0)

        task.cancel()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()

        with pytest.raises(RuntimeError, match="upsert rejected"):
            await task
        assert slow_completed is True
