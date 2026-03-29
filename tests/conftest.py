"""Shared pytest fixtures for mcp-codesearch tests."""

import pytest


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing chunking."""
    return '''"""Module docstring for testing."""

import os
from pathlib import Path

def simple_function(x, y):
    """Add two numbers."""
    return x + y


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.value = 0

    def add(self, x):
        """Add to current value."""
        self.value += x
        return self

    def subtract(self, x):
        """Subtract from current value."""
        self.value -= x
        return self


def helper_function():
    """A helper function."""
    pass
'''


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing."""
    return '''/**
 * Module for handling user operations.
 */

import { User } from "./types";

interface UserService {
    getUser(id: string): Promise<User>;
    updateUser(user: User): Promise<void>;
}

class UserServiceImpl implements UserService {
    async getUser(id: string): Promise<User> {
        // Implementation
        return { id, name: "test" };
    }

    async updateUser(user: User): Promise<void> {
        // Implementation
    }
}

export function createUserService(): UserService {
    return new UserServiceImpl();
}
'''


@pytest.fixture
def sample_rust_code():
    """Sample Rust code for testing."""
    return '''//! Crate-level documentation.
//! This module handles data processing.

use std::io;

/// A data processor struct.
pub struct Processor {
    buffer: Vec<u8>,
}

impl Processor {
    /// Create a new processor.
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Process the input data.
    pub fn process(&mut self, data: &[u8]) -> io::Result<()> {
        self.buffer.extend_from_slice(data);
        Ok(())
    }
}

fn helper() {
    // Internal helper
}
'''


@pytest.fixture
def temp_codebase(tmp_path):
    """Create a temporary codebase for integration tests."""
    # Create directory structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Python file
    (src_dir / "main.py").write_text('''"""Main module."""

def main():
    """Entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

    # TypeScript file
    (src_dir / "utils.ts").write_text('''/**
 * Utility functions.
 */

export function formatDate(date: Date): string {
    return date.toISOString();
}

export function parseNumber(s: string): number {
    return parseInt(s, 10);
}
''')

    # Tests directory
    test_dir = tmp_path / "tests"
    test_dir.mkdir()

    (test_dir / "test_main.py").write_text('''"""Tests for main module."""

def test_main():
    assert True
''')

    # Track collection name for cleanup
    from mcp_codesearch.storage.qdrant import collection_name
    col_name = collection_name(str(tmp_path.resolve()))

    yield tmp_path

    # Cleanup: Delete Qdrant collection
    import asyncio
    try:
        from mcp_codesearch.storage.qdrant import QdrantStorage
        storage = QdrantStorage()

        async def cleanup():
            try:
                await storage.delete_collection(col_name)
            except Exception:
                pass
            await storage.close()

        asyncio.run(cleanup())
    except Exception:
        pass  # Collection may not exist


@pytest.fixture
def empty_file_content():
    """Empty file content for edge case testing."""
    return ""


@pytest.fixture
def large_class_code():
    """Python class with many methods for testing class overview generation."""
    methods = "\n\n".join([
        f'''    def method_{i}(self, arg):
        """Method {i} docstring."""
        return arg * {i}'''
        for i in range(30)
    ])

    return f'''"""Large class module."""

class LargeClass:
    """A class with many methods."""

{methods}
'''


def pytest_sessionfinish(session, exitstatus):
    """Clean up orphaned collections at end of test session."""
    import asyncio

    from mcp_codesearch.storage.qdrant import QdrantStorage

    async def cleanup():
        storage = QdrantStorage()
        try:
            collections = await storage.list_collections()
            for col in collections:
                # Only clean up test collections (paths that don't exist)
                path = await storage.infer_codebase_path(col)
                if path and "pytest" in path:
                    try:
                        await storage.delete_collection(col)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            await storage.close()

    try:
        asyncio.run(cleanup())
    except Exception:
        pass
