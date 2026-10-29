"""
Unit tests for PhoneDetector class.
Tests class resolution, threshold filtering, mock inference, and annotation.
Free of emojis.
"""

import os
import sys

# Ensure parent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from detector import PhoneDetector


class TestPhoneDetector(unittest.TestCase):

    def test_resolve_phone_class_from_dict(self):
        """Test resolving phone class ID from standard COCO dictionary."""
        classes = {0: "person", 1: "bicycle", 67: "cell phone", 68: "microwave"}
        cls_id, name = PhoneDetector._resolve_phone_class(classes)
        self.assertEqual(cls_id, 67)
        self.assertEqual(name, "cell phone")

    def test_resolve_phone_class_alternative_naming(self):
        """Test resolving phone class ID from alternative class naming."""
        classes = {0: "person", 14: "mobile phone", 20: "laptop"}
        cls_id, name = PhoneDetector._resolve_phone_class(classes)
        self.assertEqual(cls_id, 14)
        self.assertEqual(name, "mobile phone")

    def test_resolve_phone_class_from_list(self):
        """Test resolving phone class ID from class names list."""
        classes = ["person", "car", "cell phone"]
        cls_id, name = PhoneDetector._resolve_phone_class(classes)
        self.assertEqual(cls_id, 2)
        self.assertEqual(name, "cell phone")

    def test_resolve_phone_class_missing_raises_error(self):
        """Test that missing phone class raises an informative ValueError."""
        classes = {0: "person", 1: "bicycle", 2: "car"}
        with self.assertRaises(ValueError) as ctx:
            PhoneDetector._resolve_phone_class(classes)
        self.assertIn("No recognizable cell phone class found", str(ctx.exception))

    @patch("detector.PhoneDetector._load_model")
    def test_mock_detection_filtering(self, mock_load):
        """Test filtering detections by phone class and confidence threshold."""
        detector = PhoneDetector.__new__(PhoneDetector)
        detector.phone_class_id = 67
        detector.phone_class_name = "cell phone"
        detector.model_name_or_path = "mock.pt"
        detector.device = "cpu"

        boxes_container = MagicMock()
        boxes_container.__len__.return_value = 3
        boxes_container.cls = [MagicMock(item=lambda: 67), MagicMock(item=lambda: 0), MagicMock(item=lambda: 67)]
        boxes_container.conf = [MagicMock(item=lambda: 0.85), MagicMock(item=lambda: 0.92), MagicMock(item=lambda: 0.25)]

        xyxy_mock = MagicMock()
        xyxy_mock.__getitem__.return_value.cpu.return_value.numpy.return_value.astype.return_value.tolist.return_value = [10, 20, 100, 150]
        boxes_container.xyxy = xyxy_mock

        mock_result = MagicMock()
        mock_result.boxes = boxes_container

        mock_model = MagicMock(return_value=[mock_result])
        detector.model = mock_model

        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = detector.detect(dummy_img, conf_threshold=0.40)

        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["class_id"], 67)
        self.assertEqual(dets[0]["confidence"], 0.85)
        self.assertEqual(dets[0]["box"], [10, 20, 100, 150])

    @patch("detector.PhoneDetector._load_model")
    def test_annotation_drawing(self, mock_load):
        """Test that annotation returns an annotated copy without mutating input."""
        detector = PhoneDetector.__new__(PhoneDetector)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{
            "box": [10, 10, 50, 50],
            "confidence": 0.88,
            "class_name": "cell phone",
            "class_id": 67
        }]

        annotated = detector.annotate(img, detections)
        self.assertEqual(annotated.shape, img.shape)
        self.assertEqual(img.sum(), 0)
        self.assertGreater(annotated.sum(), 0)


if __name__ == "__main__":
    unittest.main()
