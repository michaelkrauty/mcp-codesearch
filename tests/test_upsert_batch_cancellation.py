"""A failed batch upsert must not leave sibling writes running.

`asyncio.gather` re-raises the first failure without stopping the others. The
caller treats a raised upsert as "this batch did not land" and removes the
batch's points, so a task that outlives that cleanup puts some of them back --
leaving points the vocabulary no longer accounts for, and a file that looks
indexed to the next run.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_codesearch.storage import qdrant as qdrant_module
from mcp_codesearch.storage.qdrant import QdrantStorage


class TestUpsertBatchCancelsSiblings:
    async def test_a_failing_batch_cancels_the_others(self, monkeypatch):
        # Keep the batch timeout short: if cancellation regresses, the test
        # should fail quickly rather than block on the production default.
        monkeypatch.setattr(qdrant_module.settings, "upsert_batch_timeout", 10.0)

        storage = QdrantStorage()

        first_started = asyncio.Event()
        slow_completed = False
        call_index = 0

        async def upsert(collection, batch):
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
            await storage.upsert_batch(
                "col", points, batch_size=1, concurrency=2, max_retries=1
            )

        # Wait past the point where an uncancelled sibling would have
        # finished its write. Reaching here with it still incomplete is
        # what shows it was cancelled, rather than left to write after the
        # caller has already cleaned the batch up.
        await asyncio.sleep(1.0)
        assert slow_completed is False
