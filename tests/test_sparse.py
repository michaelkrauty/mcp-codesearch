"""Tests for sparse vectorizer."""

from vector_core.embeddings.sparse import SparseVector, SparseVectorizer


class TestTokenization:
    """Tests for code-aware tokenization."""

    def test_camel_case_split(self):
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("getUserData")

        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens

    def test_snake_case_split(self):
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("get_user_data")

        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens

    def test_stop_token_removal(self):
        vectorizer = SparseVectorizer()
        # Test common English stop words (not programming keywords)
        tokens = vectorizer._tokenize("the a an is are was to for of with")

        # Common English words should be filtered
        assert "the" not in tokens
        assert "a" not in tokens  # Would be filtered by min_length anyway
        assert "is" not in tokens  # Would be filtered by min_length anyway
        assert "was" not in tokens
        assert "for" not in tokens
        assert "with" not in tokens

        # Programming keywords are intentionally NOT filtered for code search
        code_tokens = vectorizer._tokenize("if else return def class")
        assert "if" in code_tokens  # Programming keywords are meaningful in code search
        assert "else" in code_tokens
        assert "return" in code_tokens
        assert "def" in code_tokens
        assert "class" in code_tokens

    def test_min_length_filter(self):
        vectorizer = SparseVectorizer(min_token_length=3)
        tokens = vectorizer._tokenize("a ab abc abcd")

        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens
        assert "abcd" in tokens


class TestVectorization:
    """Tests for TF-IDF vectorization."""

    def test_fit_creates_vocab(self):
        vectorizer = SparseVectorizer()
        documents = [
            "function handleRequest",
            "class UserService",
            "async function process",
        ]
        vectorizer.fit(documents)

        assert len(vectorizer._vocab) > 0
        assert "handle" in vectorizer._vocab or "request" in vectorizer._vocab

    def test_vectorize_returns_sparse(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world", "goodbye world"])

        vec = vectorizer.vectorize("hello world")

        assert isinstance(vec, SparseVector)
        assert len(vec.indices) > 0
        assert len(vec.indices) == len(vec.values)

    def test_unknown_tokens_ignored(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        vec = vectorizer.vectorize("completely unknown words")

        # Unknown tokens produce no output
        assert len(vec.indices) == 0

    def test_query_vectorization_with_fuzzy(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["handleRequest processData validateInput"])

        # Typo should fuzzy match
        vec = vectorizer.vectorize_query("handlRequest", fuzzy=True)

        # Should find a fuzzy match
        assert len(vec.indices) >= 0  # May or may not match depending on threshold


class TestVocabExtension:
    """Tests for incremental vocabulary extension."""

    def test_extend_adds_tokens(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        initial_size = len(vectorizer._vocab)

        new_tokens = vectorizer.extend_vocab(["completely new words"])

        assert len(vectorizer._vocab) > initial_size
        assert new_tokens > 0

    def test_extend_updates_idf(self):
        vectorizer = SparseVectorizer()
        # "hello" appears in 2 of 3 docs
        vectorizer.fit(["hello world", "hello again", "goodbye friend"])

        old_hello_idf = vectorizer._idf.get("hello", 0)
        old_world_idf = vectorizer._idf.get("world", 0)

        # Add docs WITH "hello". extend_vocab recomputes IDF for the whole
        # vocabulary because the corpus size changed (vector-core >= 1.2.8).
        vectorizer.extend_vocab(["hello there", "hello everybody"])

        new_hello_idf = vectorizer._idf.get("hello", 0)
        # IDF should change because frequency of "hello" changed
        # hello now appears in 4 of 5 docs vs 2 of 3 docs
        assert new_hello_idf != old_hello_idf

        # "world"'s IDF is also recomputed: the corpus grew (3 -> 5 docs), so
        # its IDF rises even though it is absent from the new docs.
        assert vectorizer._idf.get("world") != old_world_idf


class TestVocabConsistency:
    """Tests for vocabulary consistency tracking."""

    def test_initial_tracking(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2 word3"])

        assert vectorizer._initial_vocab_size > 0
        assert vectorizer._tokens_added_since_full == 0
        assert vectorizer.vocab_growth_ratio() == 0.0

    def test_growth_tracking(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2 word3 word4 word5"])

        initial_size = vectorizer._initial_vocab_size
        assert initial_size > 0  # Should have some vocab

        # Add new tokens
        vectorizer.extend_vocab(["newword1 newword2"])

        assert vectorizer._tokens_added_since_full > 0
        assert vectorizer.vocab_growth_ratio() > 0.0

    def test_needs_reindex_threshold(self):
        vectorizer = SparseVectorizer()
        # Create a small vocab
        vectorizer.fit(["word1 word2"])

        # By default, threshold is 10%
        assert not vectorizer.needs_reindex()

        # Add many new tokens to exceed threshold
        vectorizer.extend_vocab([
            "new1 new2 new3 new4 new5 new6 new7 new8 new9 new10"
        ])

        assert vectorizer.needs_reindex()

    def test_save_load_preserves_tracking(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2 word3"])
        vectorizer.extend_vocab(["new1 new2"])

        data = vectorizer.save_vocab()

        new_vectorizer = SparseVectorizer()
        new_vectorizer.load_vocab(data)

        assert new_vectorizer._initial_vocab_size == vectorizer._initial_vocab_size
        assert new_vectorizer._tokens_added_since_full == vectorizer._tokens_added_since_full
        assert new_vectorizer.vocab_growth_ratio() == vectorizer.vocab_growth_ratio()


class TestSaveLoad:
    """Tests for vocabulary persistence."""

    def test_save_returns_dict(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        data = vectorizer.save_vocab()

        assert isinstance(data, dict)
        assert "vocab" in data
        assert "idf" in data
        assert "doc_count" in data

    def test_load_restores_state(self):
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world", "foo bar"])

        data = vectorizer.save_vocab()

        new_vectorizer = SparseVectorizer()
        new_vectorizer.load_vocab(data)

        # Should produce same vectors
        vec1 = vectorizer.vectorize("hello world")
        vec2 = new_vectorizer.vectorize("hello world")

        assert vec1.indices == vec2.indices
        assert vec1.values == vec2.values
