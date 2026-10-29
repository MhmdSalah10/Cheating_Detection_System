"""
Video Processing Pipeline for Exam Recordings.
Sequentially processes video frames with pretrained YOLO detector,
annotates detected mobile phones, triggers cooldown alerts, and saves evidence.
All code, comments, and logs are strictly free of emojis.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import cv2

from detector import PhoneDetector
from alert_manager import AlertManager


class VideoProcessor:
    """
    Manages end-to-end video decoding, phone inference, frame annotation,
    evidence screenshot capture, and video encoding.
    """

    def __init__(
        self,
        detector: PhoneDetector,
        alert_manager: Optional[AlertManager] = None,
        output_dir: str = "outputs"
    ):
        """
        Initialize the video processor.
        
        Args:
            detector: Instance of PhoneDetector.
            alert_manager: Instance of AlertManager. Created automatically if omitted.
            output_dir: Base output directory.
        """
        self.detector = detector
        self.alert_manager = alert_manager or AlertManager(output_dir=output_dir)
        self.output_dir = output_dir

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extract video technical properties and validate readability.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Dictionary containing metadata: fps, width, height, frame_count, duration_seconds.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open or decode video file: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or fps != fps:
                fps = 30.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = round(frame_count / fps, 2) if frame_count > 0 else 0.0

            return {
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "frame_count": max(0, frame_count),
                "duration_seconds": duration
            }
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        conf_threshold: float = 0.40,
        stride: int = 1,
        save_screenshots: bool = True,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Process an input video frame by frame.
        
        Args:
            video_path: Path to the input video file.
            output_video_path: Destination path for the annotated video.
            conf_threshold: Confidence threshold for phone detection.
            stride: Process every N-th frame to accelerate processing on long recordings.
            save_screenshots: Whether to save screenshot images on qualifying detections.
            progress_callback: Optional function(current_frame, total_frames, stats) called periodically.
            
        Returns:
            Dictionary summarizing processing results and generated artifacts.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Target video does not exist: {video_path}")

        stride = max(1, int(stride))
        conf_threshold = float(conf_threshold)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file for processing: {video_path}")

        video_name = os.path.basename(video_path)
        video_stem = Path(video_path).stem

        screenshots_dir = os.path.join(self.output_dir, "screenshots")
        videos_dir = os.path.join(self.output_dir, "videos")
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        if output_video_path is None:
            output_video_path = os.path.join(videos_dir, f"annotated_{video_stem}.mp4")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or fps != fps:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        fourcc_options = [
            cv2.VideoWriter_fourcc(*"mp4v"),
            cv2.VideoWriter_fourcc(*"avc1"),
            cv2.VideoWriter_fourcc(*"XVID")
        ]

        for fourcc in fourcc_options:
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if writer.isOpened():
                break

        if writer is None or not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Unable to initialize VideoWriter for '{output_video_path}'. "
                "Verify codec availability on the host environment."
            )

        frame_idx = 0
        analyzed_frames = 0
        detections_count = 0
        screenshots_saved = 0
        start_time = time.time()
        cached_detections = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                video_time_seconds = frame_idx / fps

                if frame_idx % stride == 0:
                    analyzed_frames += 1
                    detections = self.detector.detect(frame, conf_threshold=conf_threshold)
                    cached_detections = detections
                else:
                    detections = cached_detections

                if detections:
                    detections_count += len(detections)
                    annotated_frame = self.detector.annotate(frame, detections)

                    if save_screenshots and (frame_idx % stride == 0):
                        if self.alert_manager.should_trigger_alert(video_time_seconds):
                            top_conf = max(d["confidence"] for d in detections)
                            clean_time = f"{video_time_seconds:.1f}".replace(".", "_")
                            sc_name = f"{video_stem}_frame_{frame_idx:06d}_{clean_time}s.jpg"
                            sc_path = os.path.join(screenshots_dir, sc_name)

                            cv2.imwrite(sc_path, annotated_frame)
                            self.alert_manager.record_alert(
                                video_name=video_name,
                                frame_number=frame_idx,
                                video_time_seconds=video_time_seconds,
                                confidence=top_conf,
                                screenshot_path=sc_path,
                                object_class="cell phone"
                            )
                            screenshots_saved += 1
                else:
                    annotated_frame = frame

                writer.write(annotated_frame)

                if progress_callback and (frame_idx % 10 == 0 or frame_idx == total_frames):
                    progress_callback(
                        frame_idx,
                        total_frames,
                        {
                            "frames_processed": frame_idx,
                            "analyzed_frames": analyzed_frames,
                            "detections_count": detections_count,
                            "screenshots_saved": screenshots_saved
                        }
                    )

        finally:
            cap.release()
            if writer:
                writer.release()

        elapsed_seconds = round(time.time() - start_time, 2)
        processing_fps = round(frame_idx / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            raise RuntimeError(
                f"Output video was not successfully written or is 0 bytes: {output_video_path}"
            )

        alert_csv_path = os.path.join(reports_dir, f"alerts_{video_stem}.csv")
        self.alert_manager.save_csv(alert_csv_path)

        summary = {
            "status": "completed",
            "video_name": video_name,
            "total_frames_read": frame_idx,
            "frames_analyzed": analyzed_frames,
            "detections_count": detections_count,
            "alerts_triggered": len(self.alert_manager.alerts),
            "screenshots_saved": screenshots_saved,
            "processing_duration_seconds": elapsed_seconds,
            "processing_fps": processing_fps,
            "output_video_path": output_video_path,
            "alert_csv_path": alert_csv_path,
            "stride": stride,
            "conf_threshold": conf_threshold
        }

        return summary
