"""
Alert Manager module for structured logging and evidence management.
Manages alert records, deduplication cooldowns, and CSV persistence.
All code, comments, and logs are strictly free of emojis.
"""

import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd


class AlertManager:
    """
    Manages detection alerts, deduplication cooldowns, and structured persistence.
    Alerts represent potentially suspicious events requiring human verification.
    """

    COLUMNS = [
        "alert_id",
        "timestamp",
        "video_name",
        "frame_number",
        "video_time_seconds",
        "object_class",
        "confidence",
        "screenshot_path"
    ]

    def __init__(self, cooldown_seconds: float = 2.0, output_dir: str = "outputs"):
        """
        Initialize the alert manager.
        
        Args:
            cooldown_seconds: Minimum seconds elapsed between saving consecutive
                              alerts/evidence screenshots for ongoing detections.
                              Note: This is temporal deduplication, not object tracking.
            output_dir: Base output directory where reports and CSVs are written.
        """
        self.cooldown_seconds = float(cooldown_seconds)
        self.output_dir = output_dir
        self.last_alert_time: Optional[float] = None
        self.alerts: List[Dict[str, Any]] = []

    def should_trigger_alert(self, video_time_seconds: float) -> bool:
        """
        Determine if an alert screenshot should be triggered based on cooldown.
        
        Args:
            video_time_seconds: Current timestamp in the video stream (seconds).
            
        Returns:
            True if sufficient time has elapsed since the last alert, False otherwise.
        """
        if self.last_alert_time is None:
            return True
        return (video_time_seconds - self.last_alert_time) >= self.cooldown_seconds

    def record_alert(
        self,
        video_name: str,
        frame_number: int,
        video_time_seconds: float,
        confidence: float,
        screenshot_path: str,
        object_class: str = "cell phone"
    ) -> Dict[str, Any]:
        """
        Create and record a structured alert for a qualifying detection.
        
        Args:
            video_name: Name of the video being analyzed.
            frame_number: Sequential index of the frame.
            video_time_seconds: Timestamp of frame in video seconds.
            confidence: Detector confidence score.
            screenshot_path: Path where the evidence screenshot is stored.
            object_class: Name of the detected class.
            
        Returns:
            The recorded alert dictionary.
        """
        self.last_alert_time = video_time_seconds
        alert_id = f"ALT-{len(self.alerts) + 1:04d}"
        iso_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        record = {
            "alert_id": alert_id,
            "timestamp": iso_timestamp,
            "video_name": os.path.basename(video_name),
            "frame_number": int(frame_number),
            "video_time_seconds": round(float(video_time_seconds), 2),
            "object_class": str(object_class),
            "confidence": round(float(confidence), 4),
            "screenshot_path": str(screenshot_path)
        }
        self.alerts.append(record)
        return record

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the alerts as a pandas DataFrame.
        """
        if not self.alerts:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(self.alerts)[self.COLUMNS]

    def save_csv(self, csv_path: str, append_existing: bool = False) -> str:
        """
        Export alert records to a CSV file.
        
        Args:
            csv_path: Destination path for the CSV file.
            append_existing: If True and CSV exists, merges existing records.
            
        Returns:
            The absolute path of the written CSV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        current_df = self.get_dataframe()

        if append_existing and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                existing_df = pd.read_csv(csv_path)
                combined = pd.concat([existing_df, current_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["video_name", "frame_number"])
                combined.to_csv(csv_path, index=False)
                return os.path.abspath(csv_path)
            except Exception:
                pass

        current_df.to_csv(csv_path, index=False)
        return os.path.abspath(csv_path)

    @classmethod
    def load_csv(cls, csv_path: str) -> pd.DataFrame:
        """
        Load an existing alert CSV into a DataFrame.
        
        Args:
            csv_path: Path to the CSV file.
            
        Returns:
            Loaded DataFrame or empty DataFrame if file not found.
        """
        if not os.path.exists(csv_path):
            return pd.DataFrame(columns=cls.COLUMNS)
        try:
            df = pd.read_csv(csv_path)
            for col in cls.COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[cls.COLUMNS]
        except Exception as e:
            raise IOError(f"Failed to read alert CSV file '{csv_path}': {str(e)}") from e

    def get_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics of recorded alerts.
        """
        count = len(self.alerts)
        if count == 0:
            return {
                "total_alerts": 0,
                "max_confidence": 0.0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "first_alert_time": None,
                "last_alert_time": None
            }

        confs = [a["confidence"] for a in self.alerts]
        return {
            "total_alerts": count,
            "max_confidence": round(max(confs), 4),
            "avg_confidence": round(sum(confs) / count, 4),
            "min_confidence": round(min(confs), 4),
            "first_alert_time": self.alerts[0]["video_time_seconds"],
            "last_alert_time": self.alerts[-1]["video_time_seconds"]
        }
