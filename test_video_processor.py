"""
Unit and integration tests for VideoProcessor class.
Tests synthetic video creation, video metadata inspection, frame processing,
and resource release. Free of emojis.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import unittest
from unittest.mock import MagicMock
import tempfile
import cv2
import numpy as np

from detector import PhoneDetector
from alert_manager import AlertManager
from video_processor import VideoProcessor


def create_synthetic_test_video(path: str, num_frames: int = 30, width: int = 320, height: int = 240, fps: float = 15.0):
    """Helper function to generate a valid synthetic test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cx = int((i / num_frames) * (width - 60)) + 30
        cy = height // 2
        cv2.circle(frame, (cx, cy), 20, (0, 200, 200), -1)
        cv2.putText(frame, f"F:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "synthetic_test.mp4")
        create_synthetic_test_video(self.test_video_path, num_frames=30, width=320, height=240, fps=15.0)

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(self.temp_dir)

    def test_metadata_extraction(self):
        """Test reading metadata from valid video."""
        mock_detector = MagicMock(spec=PhoneDetector)
        processor = VideoProcessor(detector=mock_detector, output_dir=self.temp_dir)
        meta = processor.get_video_metadata(self.test_video_path)

        self.assertEqual(meta["width"], 320)
        self.assertEqual(meta["height"], 240)
        self.assertEqual(meta["frame_count"], 30)
        self.assertAlmostEqual(meta["fps"], 15.0, delta=1.0)
        self.assertGreater(meta["duration_seconds"], 0.0)

    def test_missing_video_raises_error(self):
        """Test that non-existent video path raises FileNotFoundError."""
        mock_detector = MagicMock(spec=PhoneDetector)
        processor = VideoProcessor(detector=mock_detector, output_dir=self.temp_dir)
        with self.assertRaises(FileNotFoundError):
            processor.get_video_metadata("non_existent_file.mp4")

    def test_processing_pipeline_with_mock_detections(self):
        """Test processing synthetic video with mock phone detections."""
        mock_detector = MagicMock(spec=PhoneDetector)
        mock_detector.detect.return_value = [{
            "box": [50, 50, 120, 150],
            "confidence": 0.88,
            "class_name": "cell phone",
            "class_id": 67
        }]
        mock_detector.annotate.side_effect = lambda frame, dets: frame.copy()

        alert_mgr = AlertManager(cooldown_seconds=1.0, output_dir=self.temp_dir)
        processor = VideoProcessor(
            detector=mock_detector,
            alert_manager=alert_mgr,
            output_dir=self.temp_dir
        )

        out_video_path = os.path.join(self.temp_dir, "annotated_out.mp4")
        summary = processor.process_video(
            video_path=self.test_video_path,
            output_video_path=out_video_path,
            conf_threshold=0.40,
            stride=2
        )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["total_frames_read"], 30)
        self.assertEqual(summary["frames_analyzed"], 15)
        self.assertTrue(os.path.exists(out_video_path))
        self.assertGreater(os.path.getsize(out_video_path), 0)

        self.assertGreater(summary["alerts_triggered"], 0)
        self.assertTrue(os.path.exists(summary["alert_csv_path"]))

    def test_processing_pipeline_zero_detections(self):
        """Test processing when no phone detections occur."""
        mock_detector = MagicMock(spec=PhoneDetector)
        mock_detector.detect.return_value = []
        mock_detector.annotate.side_effect = lambda frame, dets: frame

        alert_mgr = AlertManager(cooldown_seconds=1.0, output_dir=self.temp_dir)
        processor = VideoProcessor(
            detector=mock_detector,
            alert_manager=alert_mgr,
            output_dir=self.temp_dir
        )

        out_video_path = os.path.join(self.temp_dir, "annotated_no_detections.mp4")
        summary = processor.process_video(
            video_path=self.test_video_path,
            output_video_path=out_video_path,
            conf_threshold=0.40
        )

        self.assertEqual(summary["detections_count"], 0)
        self.assertEqual(summary["alerts_triggered"], 0)
        self.assertEqual(summary["screenshots_saved"], 0)
        self.assertTrue(os.path.exists(out_video_path))
        self.assertGreater(os.path.getsize(out_video_path), 0)


if __name__ == "__main__":
    unittest.main()
