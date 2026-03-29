"""Tests for mcp-codesearch CLI entry point."""

import sys
from unittest.mock import patch

import pytest


class TestMain:
    """Tests for main function."""

    def test_help_flag(self, capsys):
        """--help flag prints help and exits."""
        from mcp_codesearch import __main__ as main_module

        with patch.object(sys, "argv", ["mcp-codesearch", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main_module.main()

            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "mcp-codesearch" in captured.out
        assert "Semantic Code Search" in captured.out
        assert "MCP TOOLS PROVIDED" in captured.out

    def test_h_flag(self, capsys):
        """-h flag prints help and exits."""
        from mcp_codesearch import __main__ as main_module

        with patch.object(sys, "argv", ["mcp-codesearch", "-h"]):
            with pytest.raises(SystemExit) as exc_info:
                main_module.main()

            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "mcp-codesearch" in captured.out

    def test_main_runs_server(self):
        """Main runs MCP server when no help flag."""
        with patch.object(sys, "argv", ["mcp-codesearch"]):
            # Mock the server.main function that gets imported inside main()
            with patch("mcp_codesearch.server.main") as mock_server_main:
                from mcp_codesearch.__main__ import main
                main()
                mock_server_main.assert_called_once()

    def test_help_text_contents(self):
        """Help text contains expected sections."""
        from mcp_codesearch.__main__ import HELP_TEXT

        # Check for all expected sections
        assert "USAGE:" in HELP_TEXT
        assert "DESCRIPTION:" in HELP_TEXT
        assert "MCP TOOLS PROVIDED:" in HELP_TEXT
        assert "SEARCH SYNTAX:" in HELP_TEXT
        assert "EXAMPLES:" in HELP_TEXT
        assert "CONFIGURATION" in HELP_TEXT

        # Check for tool names
        assert "code_search" in HELP_TEXT
        assert "force_reindex" in HELP_TEXT
        assert "index_status" in HELP_TEXT
        assert "list_collections" in HELP_TEXT
        assert "preview_index" in HELP_TEXT
        assert "delete_collection" in HELP_TEXT

        # Check for search syntax
        assert "function:" in HELP_TEXT
        assert "class:" in HELP_TEXT
        assert "path:" in HELP_TEXT
