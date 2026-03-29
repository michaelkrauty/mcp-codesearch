"""Integration tests for QdrantStorage with real Qdrant."""

import pytest
from vector_core.storage.qdrant import QdrantConnectionError

from mcp_codesearch.storage.qdrant import (
    ChunkPoint,
    FilePoint,
    QdrantStorage,
    collection_name,
)

from .conftest import requires_qdrant


@requires_qdrant
class TestQdrantStorageConnection:
    """Tests for Qdrant connection handling."""

    async def test_connection(self, qdrant_storage):
        """Can connect to Qdrant."""
        client = await qdrant_storage._get_client()
        assert client is not None

    async def test_connection_error(self):
        """Connection error on invalid URL."""
        storage = QdrantStorage(url="http://localhost:9999")
        with pytest.raises(QdrantConnectionError):
            await storage.collection_exists("test")
        await storage.close()


@requires_qdrant
class TestCollectionManagement:
    """Tests for collection CRUD operations."""

    async def test_create_collection(self, qdrant_storage, test_collection_name):
        """Create a collection."""
        await qdrant_storage.create_collection(test_collection_name)

        exists = await qdrant_storage.collection_exists(test_collection_name)
        assert exists is True

        await qdrant_storage.delete_collection(test_collection_name)

    async def test_delete_collection(self, qdrant_storage, test_collection_name):
        """Delete a collection."""
        await qdrant_storage.create_collection(test_collection_name)
        await qdrant_storage.delete_collection(test_collection_name)

        exists = await qdrant_storage.collection_exists(test_collection_name)
        assert exists is False

    async def test_collection_exists_false(self, qdrant_storage):
        """Non-existent collection returns False."""
        exists = await qdrant_storage.collection_exists("nonexistent_xyz123")
        assert exists is False

    async def test_list_collections(self, qdrant_storage, test_collection_name):
        """List codesearch collections."""
        await qdrant_storage.create_collection(test_collection_name)

        collections = await qdrant_storage.list_collections()
        assert test_collection_name in collections

        await qdrant_storage.delete_collection(test_collection_name)


@requires_qdrant
class TestCollectionName:
    """Tests for collection name generation."""

    def test_collection_name_deterministic(self):
        """Same path produces same collection name."""
        name1 = collection_name("/path/to/code")
        name2 = collection_name("/path/to/code")
        assert name1 == name2
        assert name1.startswith("codesearch_")

    def test_collection_name_trailing_slash(self):
        """Trailing slashes normalized."""
        name1 = collection_name("/path/to/code")
        name2 = collection_name("/path/to/code/")
        assert name1 == name2


@requires_qdrant
class TestPointIdGeneration:
    """Tests for point ID generation."""

    def test_point_id_deterministic(self, qdrant_storage):
        """Same inputs produce same ID."""
        id1 = qdrant_storage._point_id("file", "src/main.py")
        id2 = qdrant_storage._point_id("file", "src/main.py")
        assert id1 == id2

    def test_point_id_different_types(self, qdrant_storage):
        """Different point types produce different IDs."""
        file_id = qdrant_storage._point_id("file", "src/main.py")
        chunk_id = qdrant_storage._point_id("chunk", "src/main.py", start_line=10)
        assert file_id != chunk_id

    def test_point_id_includes_line(self, qdrant_storage):
        """Chunk IDs differ by start_line."""
        id1 = qdrant_storage._point_id("chunk", "src/main.py", start_line=10)
        id2 = qdrant_storage._point_id("chunk", "src/main.py", start_line=20)
        assert id1 != id2


@requires_qdrant
class TestFileOperations:
    """Tests for file point operations."""

    async def test_upsert_file(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Upsert a file point."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        # Verify via indexed files
        indexed = await qdrant_storage.get_indexed_files(test_collection)
        assert sample_file_point.path in indexed
        assert indexed[sample_file_point.path] == sample_file_point.file_hash

    async def test_upsert_file_updates(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Upsert updates existing file."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        # Update file
        updated_file = FilePoint(
            path=sample_file_point.path,
            abs_path=sample_file_point.abs_path,
            language=sample_file_point.language,
            file_hash="newhash999",
            summary="Updated summary",
            line_count=100,
            size_bytes=2048,
            mtime=1704153600.0,
        )
        await qdrant_storage.upsert_file(
            test_collection,
            updated_file,
            sample_dense_vector,
            sample_sparse_vector,
        )

        indexed = await qdrant_storage.get_indexed_files(test_collection)
        assert indexed[sample_file_point.path] == "newhash999"

    async def test_delete_by_path(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Delete points by path."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        await qdrant_storage.delete_by_path(test_collection, sample_file_point.path)

        indexed = await qdrant_storage.get_indexed_files(test_collection)
        assert sample_file_point.path not in indexed


@requires_qdrant
class TestChunkOperations:
    """Tests for chunk point operations."""

    async def test_upsert_chunk(
        self,
        qdrant_storage,
        test_collection,
        sample_chunk_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Upsert a chunk point."""
        await qdrant_storage.upsert_chunk(
            test_collection,
            sample_chunk_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        # Verify via hybrid search (mode=chunk)
        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            mode="chunk",
            limit=10,
        )
        # Should find at least the chunk we just inserted
        assert len(results) >= 1
        paths = [r.path for r in results]
        assert sample_chunk_point.path in paths


@requires_qdrant
class TestBatchOperations:
    """Tests for batch upsert operations."""

    async def test_upsert_batch_files(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Batch upsert multiple files."""
        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector as QdrantSparse

        points = []
        for i in range(10):
            point = PointStruct(
                id=qdrant_storage._point_id("file", f"src/file{i}.py"),
                vector={
                    "dense": sample_dense_vector,
                    "sparse": QdrantSparse(
                        indices=sample_sparse_vector.indices,
                        values=sample_sparse_vector.values,
                    ),
                },
                payload={
                    "type": "file",
                    "path": f"src/file{i}.py",
                    "abs_path": f"/project/src/file{i}.py",
                    "language": "python",
                    "file_hash": f"hash{i}",
                    "summary": f"File {i}",
                    "line_count": 50 + i,
                    "size_bytes": 1000 + i * 100,
                    "mtime": 1704067200.0 + i,
                    "indexed_at": "2024-01-01T00:00:00Z",
                },
            )
            points.append(point)

        await qdrant_storage.upsert_batch(test_collection, points)

        indexed = await qdrant_storage.get_indexed_files(test_collection)
        assert len(indexed) == 10

    async def test_upsert_batch_empty(self, qdrant_storage, test_collection):
        """Batch upsert with empty list."""
        await qdrant_storage.upsert_batch(test_collection, [])
        # Should not raise


@requires_qdrant
class TestIndexedFilesQueries:
    """Tests for indexed files queries."""

    async def test_get_indexed_files_empty(self, qdrant_storage, test_collection):
        """Empty collection returns empty dict."""
        indexed = await qdrant_storage.get_indexed_files(test_collection)
        assert indexed == {}

    async def test_get_indexed_files_metadata(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Get full metadata for indexed files."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        metadata = await qdrant_storage.get_indexed_files_metadata(test_collection)
        assert sample_file_point.path in metadata

        file_meta = metadata[sample_file_point.path]
        assert file_meta["file_hash"] == sample_file_point.file_hash
        assert file_meta["mtime"] == sample_file_point.mtime
        assert file_meta["size_bytes"] == sample_file_point.size_bytes


@requires_qdrant
class TestHybridSearch:
    """Tests for hybrid search operations."""

    async def test_hybrid_search_empty(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search on empty collection."""
        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            limit=10,
        )
        assert results == []

    async def test_hybrid_search_file_mode(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_chunk_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search in file mode only returns files."""
        # Insert both file and chunk
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )
        await qdrant_storage.upsert_chunk(
            test_collection,
            sample_chunk_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            mode="file",
            limit=10,
        )

        assert len(results) >= 1
        assert all(r.point_type == "file" for r in results)

    async def test_hybrid_search_chunk_mode(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_chunk_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search in chunk mode only returns chunks."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )
        await qdrant_storage.upsert_chunk(
            test_collection,
            sample_chunk_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            mode="chunk",
            limit=10,
        )

        assert len(results) >= 1
        assert all(r.point_type == "chunk" for r in results)

    async def test_hybrid_search_both_mode(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_chunk_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search in both mode returns all types."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )
        await qdrant_storage.upsert_chunk(
            test_collection,
            sample_chunk_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            mode="both",
            limit=10,
        )

        assert len(results) == 2
        types = {r.point_type for r in results}
        assert types == {"file", "chunk"}

    async def test_hybrid_search_language_filter(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search filters by language."""
        # Insert Python file
        py_file = FilePoint(
            path="src/main.py",
            abs_path="/project/src/main.py",
            language="python",
            file_hash="hash1",
            summary="Python file",
            line_count=50,
            size_bytes=1000,
        )
        await qdrant_storage.upsert_file(
            test_collection, py_file, sample_dense_vector, sample_sparse_vector
        )

        # Insert TypeScript file
        ts_file = FilePoint(
            path="src/main.ts",
            abs_path="/project/src/main.ts",
            language="typescript",
            file_hash="hash2",
            summary="TypeScript file",
            line_count=50,
            size_bytes=1000,
        )
        await qdrant_storage.upsert_file(
            test_collection, ts_file, sample_dense_vector, sample_sparse_vector
        )

        # Search for Python only
        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            language="python",
            limit=10,
        )

        assert len(results) == 1
        assert results[0].language == "python"

    async def test_hybrid_search_weighted_rrf(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search with custom weights."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        # Use unequal weights to trigger weighted RRF path
        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            dense_weight=0.7,
            sparse_weight=0.3,
            limit=10,
        )

        assert len(results) >= 1


@requires_qdrant
class TestExactMatchSearch:
    """Tests for exact match fallback search."""

    async def test_exact_match_in_name(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match finds query in name."""
        chunk = ChunkPoint(
            path="src/main.py",
            abs_path="/project/src/main.py",
            language="python",
            file_hash="hash",
            chunk_type="function",
            name="processUserData",
            start_line=10,
            end_line=20,
            content="def processUserData():\n    pass",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, chunk, sample_dense_vector, sample_sparse_vector
        )

        results = await qdrant_storage.exact_match_search(
            test_collection,
            "processUserData",
            mode="chunk",
            limit=10,
        )

        assert len(results) >= 1
        assert results[0].name == "processUserData"
        assert results[0].score == 3.0  # Name match gets highest score

    async def test_exact_match_in_summary(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match finds query in summary."""
        file_point = FilePoint(
            path="src/auth.py",
            abs_path="/project/src/auth.py",
            language="python",
            file_hash="hash",
            summary="Authentication module for user login",
            line_count=100,
            size_bytes=2000,
        )
        await qdrant_storage.upsert_file(
            test_collection, file_point, sample_dense_vector, sample_sparse_vector
        )

        results = await qdrant_storage.exact_match_search(
            test_collection,
            "Authentication",
            mode="file",
            limit=10,
        )

        assert len(results) >= 1
        assert "auth.py" in results[0].path
        assert results[0].score == 2.0  # Summary match

    async def test_exact_match_in_content(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match finds query in content."""
        chunk = ChunkPoint(
            path="src/utils.py",
            abs_path="/project/src/utils.py",
            language="python",
            file_hash="hash",
            chunk_type="function",
            name="helper",
            start_line=5,
            end_line=10,
            content="def helper():\n    # uniqueStringXYZ123\n    pass",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, chunk, sample_dense_vector, sample_sparse_vector
        )

        results = await qdrant_storage.exact_match_search(
            test_collection,
            "uniqueStringXYZ123",
            mode="chunk",
            limit=10,
        )

        assert len(results) >= 1
        assert results[0].score == 1.0  # Content match

    async def test_exact_match_language_filter(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match respects language filter."""
        py_chunk = ChunkPoint(
            path="src/main.py",
            abs_path="/project/src/main.py",
            language="python",
            file_hash="hash",
            chunk_type="function",
            name="findUser",
            start_line=10,
            end_line=20,
            content="def findUser(): pass",
            context=None,
        )
        ts_chunk = ChunkPoint(
            path="src/main.ts",
            abs_path="/project/src/main.ts",
            language="typescript",
            file_hash="hash2",
            chunk_type="function",
            name="findUser",
            start_line=10,
            end_line=20,
            content="function findUser() {}",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, py_chunk, sample_dense_vector, sample_sparse_vector
        )
        await qdrant_storage.upsert_chunk(
            test_collection, ts_chunk, sample_dense_vector, sample_sparse_vector
        )

        results = await qdrant_storage.exact_match_search(
            test_collection,
            "findUser",
            language="python",
            limit=10,
        )

        assert len(results) == 1
        assert results[0].language == "python"

    async def test_exact_match_word_boundary(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match uses word boundaries."""
        chunk = ChunkPoint(
            path="src/dialog.py",
            abs_path="/project/src/dialog.py",
            language="python",
            file_hash="hash",
            chunk_type="class",
            name="Dialog",
            start_line=1,
            end_line=50,
            content="class Dialog:\n    pass",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, chunk, sample_dense_vector, sample_sparse_vector
        )

        # "log" should NOT match "Dialog" (word boundary)
        results = await qdrant_storage.exact_match_search(
            test_collection,
            "log",
            mode="chunk",
            limit=10,
        )

        # Should not find Dialog when searching for "log"
        assert not any(r.name == "Dialog" for r in results)


@requires_qdrant
class TestMetadataStorage:
    """Tests for collection metadata storage."""

    async def test_store_metadata(self, qdrant_storage, test_collection):
        """Store and retrieve metadata."""
        codebase_path = "/path/to/project"

        await qdrant_storage.store_metadata(
            test_collection,
            codebase_path=codebase_path,
        )

        metadata = await qdrant_storage.get_metadata(test_collection)
        assert metadata is not None
        assert metadata["codebase_path"] == codebase_path

    async def test_get_metadata_none(self, qdrant_storage, test_collection):
        """Get metadata returns None when not set."""
        metadata = await qdrant_storage.get_metadata(test_collection)
        assert metadata is None


@requires_qdrant
class TestInferCodebasePath:
    """Tests for codebase path inference."""

    async def test_infer_codebase_path(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Infer codebase path from file points."""
        file_point = FilePoint(
            path="src/main.py",
            abs_path="/home/user/project/src/main.py",
            language="python",
            file_hash="hash",
            summary="Main module",
            line_count=50,
            size_bytes=1000,
        )
        await qdrant_storage.upsert_file(
            test_collection, file_point, sample_dense_vector, sample_sparse_vector
        )

        path = await qdrant_storage.infer_codebase_path(test_collection)
        assert path == "/home/user/project"

    async def test_infer_codebase_path_empty(self, qdrant_storage, test_collection):
        """Infer returns None on empty collection."""
        path = await qdrant_storage.infer_codebase_path(test_collection)
        assert path is None


# Note: TestWeightedRRF removed - RRF logic is now in vector-core.storage.hybrid
# and is tested in vector-core/tests/test_hybrid.py


@requires_qdrant
class TestQdrantEdgeCases:
    """Tests for edge cases to improve coverage."""

    async def test_hybrid_search_equal_weights_rrf(
        self,
        qdrant_storage,
        test_collection,
        sample_file_point,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Hybrid search with exactly equal weights uses fast RRF path (lines 454-476)."""
        await qdrant_storage.upsert_file(
            test_collection,
            sample_file_point,
            sample_dense_vector,
            sample_sparse_vector,
        )

        # Use exactly equal weights to trigger fast RRF fusion path
        # Default weights are 1.0 and 0.8, so we need to pass explicit equal weights
        results = await qdrant_storage.hybrid_search(
            test_collection,
            sample_dense_vector,
            sample_sparse_vector,
            dense_weight=1.0,
            sparse_weight=1.0,  # Equal weights trigger fast path
            limit=10,
        )

        assert len(results) >= 1

    async def test_exact_match_skip_metadata_point(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match skips __metadata__ points (line 591)."""
        # Store metadata which creates a __metadata__ point
        await qdrant_storage.store_metadata(
            test_collection,
            codebase_path="/test/path",
        )

        # Insert a real chunk
        chunk = ChunkPoint(
            path="src/test.py",
            abs_path="/project/src/test.py",
            language="python",
            file_hash="hash",
            chunk_type="function",
            name="testFunc",
            start_line=1,
            end_line=10,
            content="def testFunc(): pass",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, chunk, sample_dense_vector, sample_sparse_vector
        )

        # Use mode="both" to avoid type filter, which allows __metadata__ to be seen
        # The code should skip it internally
        results = await qdrant_storage.exact_match_search(
            test_collection,
            "testFunc",
            mode="both",  # Use 'both' to not filter by type at Qdrant level
            limit=10,
        )

        assert len(results) >= 1
        assert results[0].name == "testFunc"
        # Should not have returned metadata point as result
        assert not any(r.name == "__metadata__" for r in results)

    async def test_exact_match_large_content_fallback(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match falls back for large content (line 617).

        Note: Content is truncated to max_payload_content_chars (30k default) at storage,
        so this path with 50k+ content can only be hit with mocked data.
        This tests the defensive code path.
        """
        from unittest.mock import MagicMock, patch

        # Create a mock point with large content that would trigger line 617
        large_content = "x" * 25000 + "uniqueLargeMarker" + "x" * 26000
        mock_point = MagicMock()
        mock_point.payload = {
            "type": "chunk",
            "path": "src/big.py",
            "abs_path": "/project/src/big.py",
            "language": "python",
            "file_hash": "hash",
            "chunk_type": "function",
            "name": "otherName",
            "start_line": 1,
            "end_line": 1000,
            "content": large_content,  # Large content >50000 chars
            "context": None,
            "summary": "",
        }

        # Create a real chunk first so collection has data
        chunk = ChunkPoint(
            path="src/small.py",
            abs_path="/project/src/small.py",
            language="python",
            file_hash="hash",
            chunk_type="function",
            name="smallFunc",
            start_line=1,
            end_line=10,
            content="def smallFunc(): pass",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, chunk, sample_dense_vector, sample_sparse_vector
        )

        # Mock scroll to return point with large content
        client = await qdrant_storage._get_client()

        async def mock_scroll(*args, **kwargs):
            # First call returns mock with large content, second returns empty
            if not hasattr(mock_scroll, 'called'):
                mock_scroll.called = True
                return ([mock_point], None)
            return ([], None)

        with patch.object(client, 'scroll', side_effect=mock_scroll):
            results = await qdrant_storage.exact_match_search(
                test_collection,
                "uniqueLargeMarker",
                mode="chunk",
                limit=10,
            )

        # Should find via content match using fallback substring search
        assert len(results) >= 1
        assert results[0].content == large_content

    async def test_exact_match_malformed_regex_fallback(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match falls back on malformed regex (lines 620-622)."""
        import re
        from unittest.mock import patch

        chunk = ChunkPoint(
            path="src/test.py",
            abs_path="/project/src/test.py",
            language="python",
            file_hash="hash",
            chunk_type="function",
            name="testMethod",
            start_line=1,
            end_line=10,
            content="def testMethod(): pass",
            context=None,
        )
        await qdrant_storage.upsert_chunk(
            test_collection, chunk, sample_dense_vector, sample_sparse_vector
        )

        # Mock re.search to raise re.error to trigger fallback path
        original_search = re.search
        call_count = 0

        def mock_search(pattern, text, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First few calls fail with regex error, then succeed
            if call_count <= 3:
                raise re.error("Simulated regex error")
            return original_search(pattern, text, *args, **kwargs)

        with patch('re.search', side_effect=mock_search):
            results = await qdrant_storage.exact_match_search(
                test_collection,
                "testMethod",
                mode="chunk",
                limit=10,
            )

        # May or may not find results due to mock, just check it handled gracefully
        assert isinstance(results, list)

    async def test_exact_match_limit_early_break(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Exact match breaks early when limit reached (line 657)."""
        # Insert multiple matching chunks
        for i in range(10):
            chunk = ChunkPoint(
                path=f"src/file{i}.py",
                abs_path=f"/project/src/file{i}.py",
                language="python",
                file_hash=f"hash{i}",
                chunk_type="function",
                name="findData",  # Same name in all
                start_line=1,
                end_line=10,
                content=f"def findData{i}(): pass",
                context=None,
            )
            await qdrant_storage.upsert_chunk(
                test_collection, chunk, sample_dense_vector, sample_sparse_vector
            )

        # Search with small limit
        results = await qdrant_storage.exact_match_search(
            test_collection,
            "findData",
            mode="chunk",
            limit=3,  # Small limit to trigger early break
        )

        # Should have exactly 3 results due to limit
        assert len(results) == 3

    async def test_infer_codebase_path_exception(
        self,
        qdrant_storage,
    ):
        """infer_codebase_path returns None on exception (lines 721-722)."""
        # Call with non-existent collection should trigger exception path
        path = await qdrant_storage.infer_codebase_path("nonexistent_collection_xyz")
        assert path is None

    async def test_upsert_batch_retry_on_error(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Upsert batch retries on transient errors (lines 295-299)."""
        from unittest.mock import patch

        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector as QdrantSparse

        # Create test points
        points = [
            PointStruct(
                id=qdrant_storage._point_id("file", f"src/retry{i}.py"),
                vector={
                    "dense": sample_dense_vector,
                    "sparse": QdrantSparse(
                        indices=sample_sparse_vector.indices,
                        values=sample_sparse_vector.values,
                    ),
                },
                payload={
                    "type": "file",
                    "path": f"src/retry{i}.py",
                    "file_hash": f"hash{i}",
                },
            )
            for i in range(3)
        ]

        # Mock the client's upsert to fail first, then succeed
        call_count = 0
        original_upsert = None

        async def mock_upsert(collection, batch):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Transient error")
            # On retry, succeed
            return await original_upsert(collection, batch)

        client = await qdrant_storage._get_client()
        original_upsert = client.upsert

        with patch.object(client, 'upsert', side_effect=mock_upsert):
            await qdrant_storage.upsert_batch(test_collection, points)

        # Should have retried
        assert call_count >= 2

    async def test_upsert_batch_all_retries_fail(
        self,
        qdrant_storage,
        test_collection,
        sample_dense_vector,
        sample_sparse_vector,
    ):
        """Upsert batch raises after all retries exhausted (lines 300-301)."""
        from unittest.mock import patch

        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector as QdrantSparse

        # Create test points
        points = [
            PointStruct(
                id=qdrant_storage._point_id("file", "src/fail.py"),
                vector={
                    "dense": sample_dense_vector,
                    "sparse": QdrantSparse(
                        indices=sample_sparse_vector.indices,
                        values=sample_sparse_vector.values,
                    ),
                },
                payload={
                    "type": "file",
                    "path": "src/fail.py",
                    "file_hash": "hash",
                },
            )
        ]

        # Mock the client's upsert to always fail
        async def always_fail(*args, **kwargs):
            raise Exception("Persistent error")

        client = await qdrant_storage._get_client()

        with patch.object(client, 'upsert', side_effect=always_fail):
            with pytest.raises(Exception, match="Persistent error"):
                # Use small batch_size and concurrency to speed up test
                await qdrant_storage.upsert_batch(
                    test_collection, points, batch_size=10, concurrency=1
                )
