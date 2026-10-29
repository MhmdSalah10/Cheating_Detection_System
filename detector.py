"""
Mobile Phone Detector module using Ultralytics YOLO.
Identifies cell phone objects in images and video frames.
All code, comments, and logs are strictly free of emojis.
"""

import os
from typing import List, Dict, Any, Union, Tuple
import numpy as np
import cv2


class PhoneDetector:
    """
    Detector for identifying mobile phones using a pretrained YOLO model.
    Dynamically identifies the class index for 'cell phone' from model metadata.
    """

    def __init__(self, model_name_or_path: str = "yolov8n.pt", device: str = None):
        """
        Initialize the detector with a pretrained YOLO model.
        
        Args:
            model_name_or_path: Model name or weights path (e.g. yolov8n.pt, yolo11n.pt).
            device: Computing device ('cpu', 'cuda', '0', etc.). Defaults to auto-detection.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None
        self.phone_class_id = None
        self.phone_class_name = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the YOLO model and validates the presence of the cell phone class.
        Raises ValueError if the model cannot be loaded or lacks the required class.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics YOLO package is required. Install via: pip install ultralytics"
            ) from e

        try:
            self.model = YOLO(self.model_name_or_path)
            if self.device:
                self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_name_or_path}': {str(e)}"
            ) from e

        self.phone_class_id, self.phone_class_name = self._resolve_phone_class(self.model.names)

    @staticmethod
    def _resolve_phone_class(class_names: Union[Dict[int, str], List[str]]) -> Tuple[int, str]:
        """
        Inspects model class names dictionary or list to find the cell phone class ID.
        
        Args:
            class_names: Mapping or list of class IDs to class name strings.
            
        Returns:
            Tuple of (class_id, canonical_class_name).
            
        Raises:
            ValueError: If no recognizable phone class is found.
        """
        candidates = ["cell phone", "phone", "mobile phone", "telephone"]
        
        if isinstance(class_names, list):
            class_dict = {i: name for i, name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            class_dict = class_names
        else:
            raise ValueError(f"Unsupported class names format: {type(class_names)}")

        for cls_id, raw_name in class_dict.items():
            normalized = str(raw_name).strip().lower()
            if normalized in candidates or any(c in normalized for c in candidates):
                return int(cls_id), str(raw_name)

        available_classes = list(class_dict.values())[:10]
        raise ValueError(
            "No recognizable cell phone class found in model class list. "
            f"Available sample classes: {available_classes}. Cannot proceed."
        )

    def detect(
        self,
        image: Union[np.ndarray, str],
        conf_threshold: float = 0.40
    ) -> List[Dict[str, Any]]:
        """
        Detect mobile phones in an image or frame.
        
        Args:
            image: BGR numpy array or file path to image.
            conf_threshold: Minimum confidence score to register detection (default 0.40).
            
        Returns:
            List of detection dictionaries with keys:
            ['box', 'confidence', 'class_name', 'class_id']
            where box is [x1, y1, x2, y2] in integer pixel coordinates.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            loaded = cv2.imread(image)
            if loaded is None:
                raise ValueError(f"Failed to decode image from path: {image}")
            image = loaded

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array.")

        results = self.model(image, conf=conf_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes = first_res.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            if cls_id == self.phone_class_id and conf >= conf_threshold:
                coords = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                detections.append({
                    "box": coords,
                    "confidence": round(conf, 4),
                    "class_name": self.phone_class_name,
                    "class_id": cls_id
                })

        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        box_color: Tuple[int, int, int] = (0, 0, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw high-visibility red bounding boxes and confidence labels on an image copy.
        
        Args:
            image: Input BGR image array.
            detections: List of detection dictionaries returned by detect().
            box_color: BGR tuple for box color (default red: 0, 0, 255).
            text_color: BGR tuple for text color (default white: 255, 255, 255).
            line_thickness: Line thickness in pixels.
            
        Returns:
            Annotated BGR image copy.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_thickness)

            label = f"PHONE DETECTED | Confidence: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1

            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            box_top = max(0, y1 - label_h - baseline - 6)
            box_bottom = y1
            box_right = min(w, x1 + label_w + 8)

            cv2.rectangle(
                annotated,
                (x1, box_top),
                (box_right, box_bottom),
                box_color,
                -1
            )

            text_org = (x1 + 4, y1 - baseline - 3)
            cv2.putText(
                annotated,
                label,
                text_org,
                font,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA
            )

        return annotated
