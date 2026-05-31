"""Tests for Jupyter notebook source extraction."""

import json

from mcp_codesearch.indexer.notebook import extract_notebook_source


def _nb(cells, version=4):
    """Serialize cells into a notebook JSON string (v4 top-level or v3 worksheets)."""
    if version == 3:
        return json.dumps({"worksheets": [{"cells": cells}], "nbformat": 3})
    return json.dumps({"cells": cells, "nbformat": 4})


def test_extracts_code_cells_only():
    raw = _nb(
        [
            {"cell_type": "markdown", "source": "# Title\nsome prose"},
            {"cell_type": "code", "source": "import os\n"},
            {"cell_type": "code", "source": "def f():\n    return 1\n"},
            {"cell_type": "raw", "source": "raw stuff"},
        ]
    )
    out = extract_notebook_source(raw)
    assert "import os" in out
    assert "def f():" in out
    assert "some prose" not in out  # markdown dropped
    assert "raw stuff" not in out  # raw dropped


def test_cell_markers_use_notebook_position():
    raw = _nb(
        [
            {"cell_type": "markdown", "source": "# intro"},
            {"cell_type": "code", "source": "a = 1"},
            {"cell_type": "code", "source": "b = 2"},
        ]
    )
    out = extract_notebook_source(raw)
    # markers reflect the cell's index in the full notebook (markdown is cell 0)
    assert "# %% [cell 1]" in out
    assert "# %% [cell 2]" in out


def test_source_as_list_of_lines():
    raw = _nb([{"cell_type": "code", "source": ["x = 1\n", "y = 2\n"]}])
    out = extract_notebook_source(raw)
    assert "x = 1" in out and "y = 2" in out


def test_v3_notebook_input_field():
    # legacy v3: code lives under "input", cells nested under worksheets
    raw = _nb([{"cell_type": "code", "input": ["print('hi')\n"]}], version=3)
    out = extract_notebook_source(raw)
    assert "print('hi')" in out


def test_empty_and_whitespace_code_cells_skipped():
    raw = _nb(
        [
            {"cell_type": "code", "source": "   \n"},
            {"cell_type": "code", "source": ""},
        ]
    )
    assert extract_notebook_source(raw) == ""


def test_no_code_cells_returns_empty():
    raw = _nb([{"cell_type": "markdown", "source": "# only prose"}])
    assert extract_notebook_source(raw) == ""


def test_malformed_json_returns_empty():
    assert extract_notebook_source("{ not valid json") == ""
    assert extract_notebook_source("") == ""


def test_non_object_json_returns_empty():
    assert extract_notebook_source("[1, 2, 3]") == ""
    assert extract_notebook_source('"a string"') == ""


def test_non_string_source_handled():
    raw = _nb([{"cell_type": "code", "source": 12345}])  # invalid source type
    assert extract_notebook_source(raw) == ""
