"""Jupyter notebook (``.ipynb``) source extraction for code indexing.

Notebooks are JSON, not source, so they are reduced to the concatenated source of
their ``code`` cells before chunking. Markdown, raw, and output cells are dropped —
code search wants code. Code cells are joined with ``# %%`` "percent format" markers
so cell boundaries survive as valid, tree-sitter-ignorable comments.

Parsing is deliberately forgiving: any input that is not a notebook with code cells
yields an empty string, so a malformed or output-only notebook is silently skipped
rather than failing the indexing of an entire codebase.
"""

from __future__ import annotations

import json


def _cell_source(cell: dict) -> str:
    """Text of a cell's ``source`` (nbformat v4) or ``input`` (legacy v3 code cells).

    Either field may be a list of line strings or a single string.
    """
    raw = cell.get("source")
    if raw is None:
        raw = cell.get("input")  # nbformat v3 stores code under "input"
    if isinstance(raw, list):
        return "".join(part for part in raw if isinstance(part, str))
    if isinstance(raw, str):
        return raw
    return ""


def _collect_cells(notebook: dict) -> list:
    """Return the notebook's cells across nbformat versions.

    nbformat v4 keeps them at the top level under ``cells``; the legacy v3 layout
    nests them under ``worksheets[*].cells``.
    """
    cells = notebook.get("cells")
    if isinstance(cells, list):
        return cells
    worksheets = notebook.get("worksheets")
    if isinstance(worksheets, list):
        collected: list = []
        for sheet in worksheets:
            if isinstance(sheet, dict) and isinstance(sheet.get("cells"), list):
                collected.extend(sheet["cells"])
        return collected
    return []


def extract_notebook_source(raw: str) -> str:
    """Reduce a notebook's raw JSON to the source of its code cells.

    Args:
        raw: The raw ``.ipynb`` file contents (JSON).

    Returns:
        The concatenated source of every non-empty ``code`` cell, each preceded by a
        ``# %% [cell N]`` marker (N is the cell's position in the notebook). Returns
        an empty string if ``raw`` is not a parseable notebook or has no code.
    """
    try:
        notebook = json.loads(raw)
    except (json.JSONDecodeError, RecursionError, ValueError):
        return ""
    if not isinstance(notebook, dict):
        return ""

    parts: list[str] = []
    for index, cell in enumerate(_collect_cells(notebook)):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        if source.strip():
            parts.append(f"# %% [cell {index}]\n{source}")

    return "\n\n".join(parts)
