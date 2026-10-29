"""
Streamlit Web Dashboard for Cheating Detection and Evidence Review.
Analyzes exam recordings for mobile phone presence, generates visual alerts,
displays evidence screenshots, and exports structured CSV logs.
All UI text, labels, and logs are strictly free of emojis.
"""

import os
import sys
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from detector import PhoneDetector
    from alert_manager import AlertManager
    from video_processor import VideoProcessor
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from detector import PhoneDetector
    from alert_manager import AlertManager
    from video_processor import VideoProcessor


def main():
    st.set_page_config(
        page_title="Cheating Detection and Alert System",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Cheating Detection and Alert System for Exam Recordings")
    st.caption("Automated Computer Vision Pipeline for Suspicious Mobile Phone Detection and Review")

    st.info(
        "Important Notice: This system detects the physical presence of mobile phones in video feeds. "
        "A detected phone constitutes a potentially suspicious event requiring human proctor review. "
        "It does not automatically verify academic dishonesty or establish student intent."
    )

    st.sidebar.header("Configuration Settings")

    model_name = st.sidebar.selectbox(
        "Pretrained Model",
        ["yolov8n.pt", "yolo11n.pt", "yolov8s.pt"],
        index=0,
        help="Select a lightweight pretrained YOLO model."
    )

    conf_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Minimum detector confidence required to register a mobile phone detection."
    )

    stride = st.sidebar.slider(
        "Frame Processing Stride",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Process every N-th frame. Stride of 1 analyzes every frame. Larger stride speeds up execution on long videos."
    )

    cooldown = st.sidebar.slider(
        "Alert Cooldown (Seconds)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Minimum elapsed video time between capturing consecutive evidence screenshots."
    )

    output_base_dir = st.sidebar.text_input(
        "Output Storage Directory",
        value="outputs",
        help="Local directory where screenshots, annotated videos, and CSV logs are stored."
    )

    col_input, col_run = st.columns([2, 1])

    with col_input:
        st.subheader("1. Video Source Selection")
        source_mode = st.radio(
            "Select Video Input Method",
            ["Upload Video File", "Select Existing Video from Storage"],
            horizontal=True
        )

        video_path = None
        temp_file_path = None

        if source_mode == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Upload Exam Recording (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"]
            )
            if uploaded_file is not None:
                suffix = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    temp_file_path = tmp.name
                video_path = temp_file_path
                st.success(f"Video uploaded: {uploaded_file.name} ({round(uploaded_file.size / 1024 / 1024, 2)} MB)")
        else:
            search_dirs = [
                os.path.join(output_base_dir, "input"),
                "data/input",
                output_base_dir,
                "."
            ]
            available_videos = []
            for d in search_dirs:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.lower().endswith((".mp4", ".avi", ".mov")) and not f.startswith("annotated_"):
                            available_videos.append(os.path.join(d, f))

            available_videos = sorted(list(set(available_videos)))
            if available_videos:
                selected = st.selectbox("Select local video", available_videos)
                video_path = selected
                st.info(f"Selected video: {os.path.abspath(video_path)}")
            else:
                st.warning(
                    "No existing video files found in local storage directories. "
                    "Please upload a video file or place an MP4 in data/input."
                )

    with col_run:
        st.subheader("2. Run Analysis")
        st.write("Ready to analyze video with configured parameters:")
        st.write(f"- Confidence: {conf_threshold:.2f}")
        st.write(f"- Stride: every {stride} frame(s)")
        st.write(f"- Cooldown: {cooldown}s")

        start_btn = st.button("Start Video Analysis", type="primary", disabled=(video_path is None))

    if start_btn and video_path:
        st.markdown("---")
        st.subheader("3. Processing Progress and Results")

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        status_text.text("Initializing YOLO detector...")
        try:
            detector = PhoneDetector(model_name_or_path=model_name)
            alert_mgr = AlertManager(cooldown_seconds=cooldown, output_dir=output_base_dir)
            processor = VideoProcessor(
                detector=detector,
                alert_manager=alert_mgr,
                output_dir=output_base_dir
            )

            meta = processor.get_video_metadata(video_path)
            status_text.text(
                f"Processing video: {meta['video_name']} | Resolution: {meta['width']}x{meta['height']} | "
                f"FPS: {meta['fps']} | Total Frames: {meta['frame_count']}"
            )

            def update_progress(current, total, stats):
                pct = min(1.0, current / total) if total > 0 else 0.0
                progress_bar.progress(pct)
                status_text.text(
                    f"Frame {current}/{total} ({int(pct*100)}%) | Analyzed: {stats['analyzed_frames']} | "
                    f"Detections: {stats['detections_count']} | Evidence Screenshots: {stats['screenshots_saved']}"
                )

            summary = processor.process_video(
                video_path=video_path,
                conf_threshold=conf_threshold,
                stride=stride,
                progress_callback=update_progress
            )

            progress_bar.progress(1.0)
            status_text.text("Processing finished successfully.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Frames", summary["total_frames_read"])
            m2.metric("Analyzed Frames", summary["frames_analyzed"])
            m3.metric("Phone Detections", summary["detections_count"])
            m4.metric("Alerts Triggered", summary["alerts_triggered"])
            m5.metric("Processing Time", f"{summary['processing_duration_seconds']}s")

            st.session_state["latest_summary"] = summary
            st.session_state["latest_alerts_df"] = alert_mgr.get_dataframe()

        except Exception as e:
            status_text.text(f"Processing failed: {str(e)}")
            st.error(f"Error during video processing: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    if "latest_summary" in st.session_state:
        summary = st.session_state["latest_summary"]
        df_alerts = st.session_state["latest_alerts_df"]

        st.markdown("---")
        st.subheader("4. Inspection and Deliverables")

        tab_video, tab_screenshots, tab_table = st.tabs([
            "Annotated Video Preview",
            "Evidence Screenshots Gallery",
            "Structured Alert Records"
        ])

        with tab_video:
            out_vid = summary.get("output_video_path")
            if out_vid and os.path.exists(out_vid):
                st.write(f"Annotated video saved at: `{out_vid}` ({round(os.path.getsize(out_vid)/1024/1024, 2)} MB)")
                try:
                    with open(out_vid, "rb") as vf:
                        vid_bytes = vf.read()
                    st.video(vid_bytes)
                    st.download_button(
                        label="Download Annotated Video",
                        data=vid_bytes,
                        file_name=os.path.basename(out_vid),
                        mime="video/mp4"
                    )
                except Exception as ve:
                    st.warning(f"Video preview could not be rendered inline: {str(ve)}. Direct download remains available.")
            else:
                st.info("No annotated video available.")

        with tab_screenshots:
            screenshots_dir = os.path.join(output_base_dir, "screenshots")
            if os.path.exists(screenshots_dir):
                sc_files = [
                    os.path.join(screenshots_dir, f)
                    for f in sorted(os.listdir(screenshots_dir))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if sc_files:
                    st.write(f"Displaying {len(sc_files)} evidence screenshot(s):")
                    cols = st.columns(3)
                    for idx, sc_path in enumerate(sc_files):
                        col = cols[idx % 3]
                        with col:
                            try:
                                img = Image.open(sc_path)
                                col.image(img, caption=os.path.basename(sc_path), use_container_width=True)
                            except Exception as ie:
                                col.write(f"Failed to load image {sc_path}: {str(ie)}")
                else:
                    st.info("No evidence screenshots were triggered during this run (zero qualifying detections).")
            else:
                st.info("Screenshots directory is empty.")

        with tab_table:
            csv_path = summary.get("alert_csv_path")
            if not df_alerts.empty:
                st.dataframe(df_alerts, use_container_width=True)
                csv_data = df_alerts.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Alert Records (CSV)",
                    data=csv_data,
                    file_name="alert_records.csv",
                    mime="text/csv"
                )
            else:
                st.info("No alert records generated. Zero phone detections exceeded the threshold.")


if __name__ == "__main__":
    main()
