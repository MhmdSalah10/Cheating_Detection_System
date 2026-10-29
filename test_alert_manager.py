"""
Unit tests for AlertManager class.
Tests cooldown deduplication, alert recording, and CSV operations.
Free of emojis.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import unittest
import tempfile
import pandas as pd
from alert_manager import AlertManager


class TestAlertManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mgr = AlertManager(cooldown_seconds=2.0, output_dir=self.temp_dir)

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(self.temp_dir)

    def test_cooldown_deduplication(self):
        """Test temporal cooldown logic."""
        self.assertTrue(self.mgr.should_trigger_alert(0.0))

        self.mgr.record_alert("video.mp4", 30, 1.0, 0.85, "path/1.jpg")

        self.assertFalse(self.mgr.should_trigger_alert(1.5))
        self.assertFalse(self.mgr.should_trigger_alert(2.9))

        self.assertTrue(self.mgr.should_trigger_alert(3.0))
        self.assertTrue(self.mgr.should_trigger_alert(4.5))

    def test_record_alert_structure(self):
        """Test structure and values of recorded alert record."""
        rec = self.mgr.record_alert(
            video_name="exam_recording.mp4",
            frame_number=145,
            video_time_seconds=4.833,
            confidence=0.8921,
            screenshot_path="/storage/sc_01.jpg"
        )

        self.assertEqual(rec["alert_id"], "ALT-0001")
        self.assertEqual(rec["video_name"], "exam_recording.mp4")
        self.assertEqual(rec["frame_number"], 145)
        self.assertEqual(rec["video_time_seconds"], 4.83)
        self.assertEqual(rec["confidence"], 0.8921)
        self.assertEqual(rec["screenshot_path"], "/storage/sc_01.jpg")
        self.assertEqual(rec["object_class"], "cell phone")
        self.assertIn("T", rec["timestamp"])

    def test_csv_save_and_reload(self):
        """Test saving alert records to CSV and reading back."""
        self.mgr.record_alert("v1.mp4", 10, 0.33, 0.75, "sc1.jpg")
        self.mgr.record_alert("v1.mp4", 90, 3.00, 0.92, "sc2.jpg")

        csv_path = os.path.join(self.temp_dir, "test_alerts.csv")
        saved_path = self.mgr.save_csv(csv_path)
        self.assertTrue(os.path.exists(saved_path))

        df_loaded = AlertManager.load_csv(csv_path)
        self.assertEqual(len(df_loaded), 2)
        self.assertEqual(list(df_loaded.columns), AlertManager.COLUMNS)
        self.assertEqual(df_loaded.iloc[0]["alert_id"], "ALT-0001")
        self.assertEqual(df_loaded.iloc[1]["confidence"], 0.92)

    def test_summary_calculation(self):
        """Test summary statistics generation."""
        summary_empty = self.mgr.get_summary()
        self.assertEqual(summary_empty["total_alerts"], 0)

        self.mgr.record_alert("v1.mp4", 10, 1.0, 0.60, "sc1.jpg")
        self.mgr.record_alert("v1.mp4", 90, 4.0, 0.90, "sc2.jpg")

        summary = self.mgr.get_summary()
        self.assertEqual(summary["total_alerts"], 2)
        self.assertEqual(summary["min_confidence"], 0.60)
        self.assertEqual(summary["max_confidence"], 0.90)
        self.assertEqual(summary["avg_confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()
