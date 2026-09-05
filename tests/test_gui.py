"""
Unit tests for Module 16 — Desktop Monitoring Dashboard GUI.
"""

import unittest
import tkinter as tk
from unittest.mock import MagicMock

from src.config import ConfigManager


class TestGUIDashboard(unittest.TestCase):
    """Test suite for Desktop Dashboard GUI."""

    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide root window during headless test runs
            self.has_tk = True
        except Exception:
            self.has_tk = False

    def tearDown(self) -> None:
        if hasattr(self, "has_tk") and self.has_tk:
            try:
                self.root.update_idletasks()
                self.root.destroy()
            except Exception:
                pass


    def test_import_gui_module(self) -> None:
        """Verifies dashboard module imports successfully."""
        import gui.dashboard as db
        self.assertTrue(hasattr(db, "DashboardApp"))
        self.assertTrue(hasattr(db, "SettingsDialog"))

    def test_dashboard_initialization(self) -> None:
        """Verifies DashboardApp initializes components when Tkinter is available."""
        if not self.has_tk:
            self.skipTest("Tkinter display environment unavailable")

        import gui.dashboard as db
        app = db.DashboardApp(self.root)

        self.assertIsNotNone(app)
        self.assertFalse(app.is_streaming)
        self.assertEqual(app.lbl_global_status.cget("text"), "IDLE")

    def test_log_event(self) -> None:
        """Verifies logging events to GUI console widget."""
        if not self.has_tk:
            self.skipTest("Tkinter display environment unavailable")

        import gui.dashboard as db
        app = db.DashboardApp(self.root)
        app.log_event("Unit Test Event Message")

        log_text = app.txt_log.get("1.0", tk.END)
        self.assertIn("Unit Test Event Message", log_text)

    def test_gui_start_stop_state_transitions(self) -> None:
        """Verifies start and stop stream state transitions."""
        if not self.has_tk:
            self.skipTest("Tkinter display environment unavailable")

        import gui.dashboard as db
        app = db.DashboardApp(self.root)

        # Mock worker thread loop to avoid starting actual video capture
        app._worker_loop = MagicMock()

        app.start_stream()
        self.assertTrue(app.is_streaming)
        self.assertEqual(app.btn_start.cget("state"), tk.DISABLED)

        app.stop_stream()
        self.assertFalse(app.is_streaming)
        self.assertEqual(app.btn_start.cget("state"), tk.NORMAL)


if __name__ == "__main__":
    unittest.main()
