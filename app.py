import base64
import io
import os
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# FloodWatch Drone - Streamlit App
# AI-powered river waste detection and flood risk analysis
# Model: Roboflow Universe trash-in-river/2
# =========================================================

DEFAULT_MODEL_ID = "trash-in-river/2"
DEFAULT_ROBOFLOW_API_KEY = "cTFdpGqaosygnJl7JMbH"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

st.set_page_config(
    page_title="FloodWatch Drone",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main {background-color: #f6faf8;}
    .block-container {padding-top: 1.5rem;}
    .hero-card {
        background: linear-gradient(135deg, #0b3d2e 0%, #147a5c 50%, #35b779 100%);
        border-radius: 24px;
        padding: 28px;
        color: white;
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        margin-bottom: 20px;
    }
    .hero-title {font-size: 34px; font-weight: 800; margin-bottom: 8px;}
    .hero-subtitle {font-size: 17px; opacity: .95; max-width: 950px;}
    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 18px;
        border: 1px solid #dcefe7;
        box-shadow: 0 8px 22px rgba(0,0,0,0.05);
    }
    .risk-low {color: #1b7f3a; font-weight: 800;}
    .risk-medium {color: #b07d00; font-weight: 800;}
    .risk-high {color: #d65a00; font-weight: 800;}
    .risk-critical {color: #c62828; font-weight: 800;}
    .small-note {font-size: 13px; color: #586069;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🚁 FloodWatch Drone</div>
        <div class="hero-subtitle">
            AI-powered UAV application for river waste detection and early flood risk analysis. 
            Designed for Sidoarjo delta-region monitoring and MIIX 2026 innovation presentation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Roboflow API Key",
    value=os.getenv("ROBOFLOW_API_KEY", DEFAULT_ROBOFLOW_API_KEY),
    type="password",
    help="API key is pre-filled. You can override it by setting ROBOFLOW_API_KEY in your environment.",
)
model_id = st.sidebar.text_input("Model ID", value=DEFAULT_MODEL_ID)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.35, 0.05)
overlap_threshold = st.sidebar.slider("Overlap / IoU Threshold", 0.05, 0.95, 0.30, 0.05)
process_every_n_frames = st.sidebar.slider("Video: Process Every N Frames", 1, 30, 8)
max_realtime_frames = st.sidebar.slider("Realtime Demo: Max Frames", 20, 500, 120, 10)

st.sidebar.markdown("---")
nst_sidebar_info = """
**Input Modes**
- Image upload
- Video upload
- Realtime camera/drone stream

**Drone stream examples**
- Webcam index: `0`
- RTSP: `rtsp://user:pass@ip:554/stream1`
- HTTP/MJPEG: `http://ip:8080/video`
"""
st.sidebar.info(nst_sidebar_info)

# -----------------------------
# Helper functions
# -----------------------------
def pil_to_base64_jpeg(image: Image.Image) -> str:
    """Convert PIL image to base64-encoded JPEG string."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def infer_roboflow(image: Image.Image, api_key: str, model_id: str, confidence: float, overlap: float) -> dict:
    """Send image to Roboflow serverless inference API."""
    if not api_key:
        raise ValueError("Roboflow API key is required.")

    image_b64 = pil_to_base64_jpeg(image)
    url = f"{ROBOFLOW_API_URL}/{model_id}"
    params = {
        "api_key": api_key,
        "confidence": int(confidence * 100),
        "overlap": int(overlap * 100),
    }
    response = requests.post(
        url,
        params=params,
        data=image_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def draw_predictions(image: Image.Image, predictions: list) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for pred in predictions:
        x = float(pred.get("x", 0))
        y = float(pred.get("y", 0))
        w = float(pred.get("width", 0))
        h = float(pred.get("height", 0))
        label = str(pred.get("class", "trash"))
        conf = float(pred.get("confidence", 0))

        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        # Thick rectangle without forcing a custom color palette
        for offset in range(3):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline="red")

        text = f"{label} {conf:.2f}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 20)), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, max(0, y1 - 20)), text, fill="white", font=font)
    return img


def predictions_to_dataframe(predictions: list) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(predictions, start=1):
        rows.append({
            "No": i,
            "Class": p.get("class", "trash"),
            "Confidence": round(float(p.get("confidence", 0)), 4),
            "X": round(float(p.get("x", 0)), 2),
            "Y": round(float(p.get("y", 0)), 2),
            "Width": round(float(p.get("width", 0)), 2),
            "Height": round(float(p.get("height", 0)), 2),
        })
    return pd.DataFrame(rows)


def calculate_risk(predictions: list, image_size: tuple) -> dict:
    """
    Simple risk estimation for demo/prototype purposes.
    Risk is based on detected waste count and estimated waste coverage area.
    """
    img_w, img_h = image_size
    image_area = max(img_w * img_h, 1)
    total_box_area = 0
    max_conf = 0

    for p in predictions:
        w = float(p.get("width", 0))
        h = float(p.get("height", 0))
        conf = float(p.get("confidence", 0))
        total_box_area += max(w * h, 0)
        max_conf = max(max_conf, conf)

    waste_count = len(predictions)
    coverage_percent = min((total_box_area / image_area) * 100, 100)

    # Prototype scoring formula. Can be improved with river segmentation, water-level sensors, rain data, GPS, and bridge/drainage proximity.
    score = min((waste_count * 8) + (coverage_percent * 2.5) + (max_conf * 10), 100)

    if score < 20:
        level = "Low"
        recommendation = "Routine monitoring. No immediate action required."
        css_class = "risk-low"
    elif score < 45:
        level = "Medium"
        recommendation = "Schedule river cleaning and monitor after rainfall."
        css_class = "risk-medium"
    elif score < 70:
        level = "High"
        recommendation = "Prioritize cleaning. Possible flow obstruction detected."
        css_class = "risk-high"
    else:
        level = "Critical"
        recommendation = "Immediate field inspection and emergency cleaning are recommended."
        css_class = "risk-critical"

    return {
        "waste_count": waste_count,
        "coverage_percent": coverage_percent,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation,
        "css_class": css_class,
    }


def show_result(original: Image.Image, annotated: Image.Image, predictions: list, risk: dict):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Input")
        st.image(original, use_column_width=True)
    with col2:
        st.subheader("Detection Result")
        st.image(annotated, use_column_width=True)

    st.markdown("### Flood Risk Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected Objects", risk["waste_count"])
    m2.metric("Estimated Coverage", f"{risk['coverage_percent']:.2f}%")
    m3.metric("Risk Score", f"{risk['risk_score']:.1f}/100")
    m4.markdown(
        f"<div class='metric-card'><b>Risk Level</b><br><span class='{risk['css_class']}'>{risk['risk_level']}</span></div>",
        unsafe_allow_html=True,
    )
    st.success(risk["recommendation"])

    df = predictions_to_dataframe(predictions)
    st.markdown("### Detection Details")
    if len(df) > 0:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Detection CSV",
            data=csv,
            file_name=f"floodwatch_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No waste objects detected at the selected confidence threshold.")


def process_image(image: Image.Image):
    with st.spinner("Running AI detection..."):
        result = infer_roboflow(image, api_key, model_id, confidence_threshold, overlap_threshold)
    predictions = result.get("predictions", [])
    annotated = draw_predictions(image, predictions)
    risk = calculate_risk(predictions, image.size)
    show_result(image, annotated, predictions, risk)

# -----------------------------
# Main tabs
# -----------------------------
tab_image, tab_video, tab_realtime, tab_about = st.tabs([
    "🖼️ Image Detection",
    "🎞️ Video Detection",
    "📡 Realtime Drone/Camera",
    "ℹ️ About System",
])

with tab_image:
    st.header("Image-Based River Waste Detection")
    uploaded_image = st.file_uploader("Upload river/drone image", type=["jpg", "jpeg", "png", "webp"])
    camera_image = st.camera_input("Or capture a quick camera image")

    image_source = uploaded_image or camera_image
    if image_source is not None:
        img = Image.open(image_source).convert("RGB")
        process_image(img)
    else:
        st.info("Upload an image or capture a camera image to start detection.")

with tab_video:
    st.header("Video-Based River Waste Detection")
    st.caption("For efficiency, the app runs inference every N frames based on the sidebar setting.")
    uploaded_video = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0
        last_predictions = []
        risk_log = []

        start = st.button("▶️ Run Video Detection")
        if start:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                if frame_idx % process_every_n_frames == 0:
                    try:
                        result = infer_roboflow(pil_frame, api_key, model_id, confidence_threshold, overlap_threshold)
                        last_predictions = result.get("predictions", [])
                    except Exception as e:
                        st.error(f"Inference error on frame {frame_idx}: {e}")
                        break

                annotated = draw_predictions(pil_frame, last_predictions)
                risk = calculate_risk(last_predictions, pil_frame.size)
                risk_log.append({
                    "Frame": frame_idx,
                    "Detected Objects": risk["waste_count"],
                    "Coverage (%)": round(risk["coverage_percent"], 2),
                    "Risk Score": round(risk["risk_score"], 2),
                    "Risk Level": risk["risk_level"],
                })

                frame_placeholder.image(annotated, channels="RGB", use_column_width=True)
                info_placeholder.markdown(
                    f"**Frame:** {frame_idx}/{total_frames} | **Risk:** {risk['risk_level']} | **Score:** {risk['risk_score']:.1f}/100"
                )
                progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            if risk_log:
                st.markdown("### Video Risk Log")
                risk_df = pd.DataFrame(risk_log)
                st.dataframe(risk_df, use_column_width=True)
                st.download_button(
                    "⬇️ Download Video Risk Log CSV",
                    data=risk_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"floodwatch_video_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload a video file to run detection.")

with tab_realtime:
    st.header("Realtime Drone / Camera Stream")
    st.write("Use this mode for a webcam, IP camera, or drone stream URL. For a webcam, enter `0`.")
    stream_source = st.text_input("Camera / Drone Stream Source", value="0")
    delay = st.slider("Display Delay per Frame (seconds)", 0.00, 1.00, 0.05, 0.01)

    col_start, col_note = st.columns([1, 2])
    with col_start:
        run_stream = st.button("📡 Start Realtime Detection")
    with col_note:
        st.markdown("<div class='small-note'>Tip: For drone demo, use RTSP/HTTP stream from the drone controller or phone bridge.</div>", unsafe_allow_html=True)

    if run_stream:
        source = 0 if stream_source.strip() == "0" else stream_source.strip()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error("Cannot open camera/drone stream. Check the source URL, network, or camera permission.")
        else:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            last_predictions = []

            for frame_idx in range(1, max_realtime_frames + 1):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Stream ended or frame could not be read.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                if frame_idx % process_every_n_frames == 0:
                    try:
                        result = infer_roboflow(pil_frame, api_key, model_id, confidence_threshold, overlap_threshold)
                        last_predictions = result.get("predictions", [])
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        break

                annotated = draw_predictions(pil_frame, last_predictions)
                risk = calculate_risk(last_predictions, pil_frame.size)

                frame_placeholder.image(annotated, channels="RGB", use_column_width=True)
                status_placeholder.markdown(
                    f"**Realtime Frame:** {frame_idx} | **Detected:** {risk['waste_count']} | "
                    f"**Risk:** <span class='{risk['css_class']}'>{risk['risk_level']}</span> | "
                    f"**Score:** {risk['risk_score']:.1f}/100",
                    unsafe_allow_html=True,
                )
                time.sleep(delay)

            cap.release()
            st.success("Realtime demo session finished. Increase max frames in sidebar for longer monitoring.")

with tab_about:
    st.header("About FloodWatch Drone")
    st.markdown(
        """
        **FloodWatch Drone** is a prototype application for detecting waste accumulation in rivers using aerial imagery.
        The system is designed to support early flood risk analysis in Sidoarjo, Indonesia, a delta region with many waterways.

        **Main Components:**
        1. Drone/camera image acquisition  
        2. Roboflow YOLO-based object detection model  
        3. Waste-density and risk-score calculation  
        4. Visual dashboard and CSV reporting  

        **Important Note:**  
        The flood risk score in this prototype is a decision-support indicator, not an official hydrological prediction.
        For operational deployment, it should be combined with river segmentation, GPS, rainfall data, water-level sensors,
        drainage maps, bridge locations, and expert validation.
        """
    )

    st.markdown("### Prototype Risk Formula")
    st.code(
        "Risk Score = (Detected Waste Count × 8) + (Estimated Waste Coverage × 2.5) + (Max Confidence × 10)",
        language="text",
    )
