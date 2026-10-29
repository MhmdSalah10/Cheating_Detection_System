# Cheating Detection and Alert System for Online and Recorded Exams

## 1. Project Title
Cheating Detection and Alert System for Online and Recorded Exams

## 2. Project Overview
The Cheating Detection and Alert System is an automated computer vision solution designed to review recorded and live exam sessions. The system analyzes exam recordings to identify potentially suspicious mobile phone presence, generates visual overlays, captures timestamped evidence screenshots, logs structured alert records, and provides an interactive Streamlit dashboard for human proctor review.

## 3. Problem Statement
Online remote examinations and unproctored video submissions present severe challenges for academic integrity. Manual review of hundreds of hours of recorded exam footage is labor-intensive, error-prone, and unsustainable for educational institutions. The unauthorized use of mobile phones is among the most frequent forms of academic dishonesty in modern digital exams. Automated visual assistance can flag moments of interest, drastically reducing human audit time while maintaining fair and accountable verification.

## 4. Objectives
- Provide automated, frame-by-frame analysis of exam video recordings.
- Detect mobile phones using lightweight, reliable pretrained object detection models.
- Apply high-visibility red bounding boxes and confidence score annotations.
- Capture deduplicated evidence screenshots with a configurable cooldown to prevent alert flooding.
- Export structured, chronological alert records to CSV for auditing.
- Provide a responsive Streamlit dashboard for video upload, execution control, evidence inspection, and report export.
- Ensure seamless deployment and persistence on Google Colab with Google Drive integration.

## 5. Technical Stack
- Language: Python 3.9+
- Object Detection: Ultralytics YOLO (YOLOv8n / YOLO11n pretrained weights)
- Computer Vision: OpenCV (cv2) for frame manipulation, decoding, and video encoding
- Web Application: Streamlit for interactive dashboard and reporting
- Data Handling: Pandas for tabular record management and CSV serialization
- Image Handling: Pillow (PIL) and NumPy
- Testing & Quality: Unittest / Pytest for automated unit and integration verification
- Cloud Platform: Google Colab with Google Drive mounting for persistence

## 6. System Architecture
The system consists of four decoupled, modular components:

```
[ Input Exam Video ]
         |
         v
[ VideoProcessor (OpenCV) ] <---> [ PhoneDetector (Ultralytics YOLO) ]
         |                                  |
         | (Sequential Frames)              | (Detections & Confidences)
         v                                  v
[ Frame Annotation ] ------------> [ AlertManager (Cooldown Logic) ]
         |                                  |
         +----------------------------------+
         |                                  |
         v                                  v
[ Annotated Video MP4 ]          [ Evidence Screenshots & CSV Log ]
         \                                  /
          \                                /
           v                              v
           [ Streamlit Web Dashboard (app.py) ]
```

1. PhoneDetector (detector.py): Encapsulates YOLO model loading, dynamically resolves the 'cell phone' class ID from model metadata, runs inference, and draws red bounding boxes with confidence labels.
2. AlertManager (alert_manager.py): Manages structured alert logs, implements temporal cooldown deduplication to prevent redundant alerts, and handles CSV persistence.
3. VideoProcessor (video_processor.py): Reads video metadata, streams frames sequentially into memory, applies detection and annotation, triggers screenshot saves, writes encoded MP4 output, and handles resource cleanup safely.
4. Streamlit Dashboard (app.py): Exposes parameter controls, progress visualization, video preview, screenshot gallery, and CSV export.

## 7. How the Pretrained Detector Works
The system utilizes pretrained YOLO models (such as YOLOv8n or YOLO11n) trained on the Microsoft COCO dataset. COCO includes 80 general object classes, one of which is 'cell phone'.
Instead of hardcoding class ID 67, PhoneDetector dynamically scans model.names to identify the numeric ID matching 'cell phone'. When an image or video frame is passed to the detector:
1. The model performs forward inference across the frame.
2. Bounding boxes are filtered for the resolved cell phone class ID and evaluated against the user-specified confidence threshold (default 0.40).
3. Qualifying detections return coordinate bounding boxes [x1, y1, x2, y2], detection confidence score, class name, and class ID.
4. The annotate() method draws high-contrast red rectangles (0, 0, 255) with an opaque label header 'PHONE DETECTED | Confidence: X.XX' in crisp white text.

## 8. How Video Processing Works
To operate efficiently in resource-constrained environments like Google Colab, the video processor operates strictly on sequential frame streams:
1. Metadata Inspection: Retrieves FPS, frame dimensions, and total frame count. Fallbacks ensure standard 30.0 FPS if metadata is corrupted or missing.
2. Sequential Iteration: Reads frames one by one via cv2.VideoCapture.read(), avoiding storing full video arrays in system RAM.
3. Configurable Stride: On long recordings, users can evaluate every N-th frame (e.g., stride=2 or stride=5) to accelerate throughput while carrying forward bounding box annotations.
4. Codec Fallback: VideoWriter attempts encoding via mp4v, falling back to avc1 or XVID if required by the operating environment.
5. Guaranteed Resource Cleanup: Video capture and writer objects are released in a finally block to prevent file corruption.

## 9. Alert and Screenshot Generation
When a phone detection exceeds the confidence threshold:
1. Cooldown Check: AlertManager evaluates the current video time against last_alert_time. If (current_time - last_alert_time) >= cooldown_seconds (default 2.0s), an alert is authorized.
2. Evidence Screenshot: The annotated frame showing the phone bounding box and label is saved to outputs/screenshots/ with a filesystem-safe, timestamped filename: {video_stem}_frame_{frame_idx:06d}_{time_seconds}s.jpg.
3. Structured Logging: A record is appended containing alert_id, UTC ISO timestamp, video filename, frame number, video timestamp in seconds, object class, confidence, and screenshot path.
4. Tabular Export: Records are exported to outputs/reports/alerts_{video_stem}.csv.

## 10. Project Directory Structure
```
Cheating_Detection_System/
|-- app.py                    # Streamlit web application
|-- detector.py               # PhoneDetector YOLO wrapper
|-- video_processor.py        # VideoProcessor pipeline
|-- alert_manager.py          # AlertManager and deduplication
|-- requirements.txt          # Python dependencies
|-- README.md                 # Complete documentation
|-- .gitignore                # Git exclusions
|-- tests/                    # Automated test suite
|   |-- test_detector.py
|   |-- test_alert_manager.py
|   |-- test_video_processor.py
|-- data/
|   |-- input/                # User exam video recordings
|   |-- test_images/          # Sample images for detector checks
|-- outputs/
|   |-- screenshots/          # Evidence screenshots
|   |-- videos/               # Annotated video recordings
|   |-- reports/              # Alert CSV logs and test reports
|   |-- logs/                 # Processing diagnostic logs
`-- models/                   # Local YOLO weights cache
```

## 11. Installation Instructions
To install and run the system locally:

1. Clone or extract the project directory:
   ```bash
   cd Cheating_Detection_System
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```
3. Install required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 12. How to Run the Streamlit App
Launch the interactive dashboard locally:
```bash
streamlit run app.py
```
Open your web browser at http://localhost:8501. Use the sidebar to configure confidence thresholds and cooldown periods, upload an exam video, and view real-time analysis results.

## 13. How to Use the Google Colab Notebook
1. Open Google Colab (https://colab.research.google.com).
2. Upload Cheating_Detection_System_Colab.ipynb.
3. Select Runtime -> Run all.
4. When prompted, authorize Google Drive access to enable persistent storage under /content/drive/MyDrive/Cheating_Detection_System/.
5. The notebook will automatically execute the staged workflow:
   - Environment check and dependency installation
   - Generation of all project source code and tests
   - Image detection checkpoint
   - Video pipeline smoke test
   - Exam video processing (supports uploaded videos or sample files)
   - Streamlit launch validation
   - Unit test execution
   - Packaging and automatic download of final_delivery.zip

## 14. Required Input Formats
- Video Formats: MP4, AVI, MOV.
- Recommended Resolution: 640x480 up to 1920x1080 (standard webcam or proctoring angles).
- Recommended Framerate: 15 to 30 FPS.
- Image Formats (for image testing): JPG, JPEG, PNG.

## 15. Configuration Options
- Confidence Threshold: Float between 0.10 and 0.90 (default: 0.40). Lower values detect subtle or distant phones; higher values reduce false positives.
- Frame Stride: Integer between 1 and 10 (default: 1). Stride 1 analyzes every frame; stride 2 analyzes every 2nd frame (2x speedup).
- Alert Cooldown: Float between 0.5 and 10.0 seconds (default: 2.0s). Dictates minimum separation between saved evidence screenshots during sustained phone visibility.
- Output Directory: Path for generated assets (default: outputs).

## 16. Example Output Descriptions
- Annotated Video: outputs/videos/annotated_exam_sample.mp4 containing visual red boxes around detected phones and confidence text.
- Evidence Screenshot: outputs/screenshots/exam_sample_frame_000142_4_7s.jpg preserving high-resolution visual context of the detection event.
- Alert CSV: outputs/reports/alerts_exam_sample.csv with columns:
  alert_id, timestamp, video_name, frame_number, video_time_seconds, object_class, confidence, screenshot_path.
- Final Archive: final_delivery.zip containing all source code, test reports, logs, sample screenshots, and documentation.

## 17. Testing Methodology and Actual Results
The project includes automated test coverage in the tests/ directory:
- test_detector.py: Validates class name lookup logic across diverse class mappings, mock inference filtering, threshold boundaries, and non-destructive image annotation.
- test_alert_manager.py: Tests temporal cooldown behavior across varying time steps, record field integrity, and CSV serialization/deserialization.
- test_video_processor.py: Uses synthetic OpenCV video streams to verify video decoding, sequential frame iteration, output file integrity, non-detection handling, and resource deallocation.
- In-Colab Stage Checks: Stages 3 and 4 enforce mandatory runtime validation (image inference and synthetic video smoke test) before attempting full exam video processing.

## 18. Known Limitations
- The detector relies on visible line-of-sight. Fully concealed devices, phones hidden beneath paper, or devices outside the webcam frame cannot be detected.
- Objects with similar rectangular form factors, such as calculators, small notebooks, or remote controls, may occasionally trigger false positives depending on lighting and angle.
- The model detects object presence, not user interaction. A phone resting untouched on a desk or an incoming notification will be flagged equally.
- Performance on Google Colab depends on GPU availability. On CPU runtimes, frame processing rate will be lower (approx. 5-10 FPS vs 40-70 FPS on GPU).

## 19. Future Improvements
- Multi-camera and multi-angle aggregation for proctored environments.
- Integration of gaze and head-pose estimation to correlate device presence with head direction.
- On-device lightweight edge deployment using ONNX Runtime or TensorRT.
- Audio event correlation to detect phone vibrations or ringtones.

## 20. Ethical Considerations and Neutral Language
- This system is an assistive review tool, not an autonomous judicial agent.
- UI labels, documentation, and logs use objective terminology: 'Potentially suspicious event', 'Phone detected', and 'Requires human review'.
- The system must never be used to make automated disciplinary decisions. Human educators and proctors must inspect evidence screenshots and contextual exam policies before taking action.
- Video recordings should be handled in compliance with student privacy regulations (FERPA, GDPR) and deleted according to institutional retention guidelines.
