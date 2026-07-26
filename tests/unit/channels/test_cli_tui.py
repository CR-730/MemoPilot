from __future__ import annotations

from textual.widgets import Static

from memopilot.channels.cli_tui import CLITextualApp


async def test_tui_keeps_migrated_header_footer_and_connection_state() -> None:
    app = CLITextualApp("127.0.0.1:1")

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        assert str(app.query_one("#title", Static).render()) == (
            "# memopilot Agent CLI / Textual"
        )
        assert str(app.query_one("#footer-left", Static).render()) == (
            "memopilot (Textual TUI)"
        )
        assert "connected: no" in str(app.query_one("#meta", Static).render())
