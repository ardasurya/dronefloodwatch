import base64
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# FloodWatch Drone - Streamlit Community Cloud Safe Version
# Image + Camera Snapshot (NO OpenCV required)
# =========================================================

DEFAULT_MODEL_ID = "trash-in-river/2"
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🚁 FloodWatch Drone</div>
        <div class="hero-subtitle">
            AI-powered river waste detection and flood-risk decision support
            using drone imagery, YOLO/Roboflow, Python, and Streamlit.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Configuration
# -----------------------------
st.sidebar.header("⚙️ Configuration")

try:
    secret_api_key = st.secrets["ROBOFLOW_API_KEY"]
except Exception:
    secret_api_key = ""

api_key = st.sidebar.text_input(
    "Roboflow API Key",
    value=secret_api_key,
    type="password",
    help="Recommended: save ROBOFLOW_API_KEY in Streamlit Secrets.",
)

model_id = st.sidebar.text_input("Model ID", value=DEFAULT_MODEL_ID)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.05, 0.95, 0.35, 0.05
)
overlap_threshold = st.sidebar.slider(
    "Overlap / IoU Threshold", 0.05, 0.95, 0.30, 0.05
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
**Cloud-safe demo modes**
- Image upload
- Browser camera snapshot

This version intentionally does not use OpenCV.
"""
)

# -----------------------------
# Helpers
# -----------------------------
def pil_to_base64_jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def infer_roboflow(
    image: Image.Image,
    api_key: str,
    model_id: str,
    confidence: float,
    overlap: float,
) -> dict:
    if not api_key:
        raise ValueError(
            "Roboflow API key is missing. Add it in the sidebar "
            "or Streamlit App Settings → Secrets."
        )

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
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
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

        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline="red",
            )

        text = f"{label} {conf:.2f}"
        tx = max(0, x1)
        ty = max(0, y1 - 22)

        try:
            bbox = draw.textbbox((tx, ty), text, font=font)
            draw.rectangle(bbox, fill="red")
        except Exception:
            pass

        draw.text((tx, ty), text, fill="white", font=font)

    return img


def predictions_to_dataframe(predictions: list) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(predictions, start=1):
        rows.append(
            {
                "No": i,
                "Class": p.get("class", "trash"),
                "Confidence": round(float(p.get("confidence", 0)), 4),
                "X": round(float(p.get("x", 0)), 2),
                "Y": round(float(p.get("y", 0)), 2),
                "Width": round(float(p.get("width", 0)), 2),
                "Height": round(float(p.get("height", 0)), 2),
            }
        )
    return pd.DataFrame(rows)


def calculate_risk(predictions: list, image_size: tuple) -> dict:
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

    # Prototype / demonstration score, not an official hydrological model.
    score = min(
        (waste_count * 8)
        + (coverage_percent * 2.5)
        + (max_conf * 10),
        100,
    )

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
        recommendation = (
            "Immediate field inspection and emergency cleaning are recommended."
        )
        css_class = "risk-critical"

    return {
        "waste_count": waste_count,
        "coverage_percent": coverage_percent,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation,
        "css_class": css_class,
    }


def show_result(
    original: Image.Image,
    annotated: Image.Image,
    predictions: list,
    risk: dict,
):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Input")
        st.image(original, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(annotated, use_container_width=True)

    st.markdown("### Flood Risk Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected Objects", risk["waste_count"])
    m2.metric("Estimated Coverage", f"{risk['coverage_percent']:.2f}%")
    m3.metric("Risk Score", f"{risk['risk_score']:.1f}/100")
    m4.markdown(
        f"""
        <div class='metric-card'>
            <b>Risk Level</b><br>
            <span class='{risk["css_class"]}'>{risk["risk_level"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.success(risk["recommendation"])

    df = predictions_to_dataframe(predictions)
    st.markdown("### Detection Details")

    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download Detection CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=(
                "floodwatch_detection_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv",
        )
    else:
        st.info("No waste objects detected at the selected confidence threshold.")


def process_image(image: Image.Image):
    try:
        with st.spinner("Running AI detection..."):
            result = infer_roboflow(
                image,
                api_key,
                model_id,
                confidence_threshold,
                overlap_threshold,
            )

        predictions = result.get("predictions", [])
        annotated = draw_predictions(image, predictions)
        risk = calculate_risk(predictions, image.size)
        show_result(image, annotated, predictions, risk)

    except requests.exceptions.Timeout:
        st.error("Roboflow inference timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Roboflow API returned an HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Network/API error: {e}")
    except Exception as e:
        st.error(f"Application error: {e}")


# -----------------------------
# Main UI
# -----------------------------
tab_detection, tab_about = st.tabs(
    ["🖼️ AI Detection", "ℹ️ About System"]
)

with tab_detection:
    st.header("River Waste Detection")

    input_mode = st.radio(
        "Choose input",
        ["Upload Image", "Camera Snapshot"],
        horizontal=True,
    )

    image_source = None

    if input_mode == "Upload Image":
        image_source = st.file_uploader(
            "Upload river/drone image",
            type=["jpg", "jpeg", "png", "webp"],
        )
    else:
        image_source = st.camera_input(
            "Capture an image using your browser camera"
        )

    if image_source is not None:
        img = Image.open(image_source).convert("RGB")

        if st.button(
            "🚀 Run AI Detection",
            type="primary",
            use_container_width=True,
        ):
            process_image(img)
    else:
        st.info("Provide an image to start AI detection.")

with tab_about:
    st.header("About FloodWatch Drone")
    st.markdown(
        """
**FloodWatch Drone** is a prototype that demonstrates how AI object
detection can support river-waste monitoring from drone or camera imagery.

### AI workflow

`Image → Streamlit → Roboflow/YOLO → Detection → Risk Indicator → Dashboard`

### Main components

1. Image acquisition
2. YOLO-based waste detection through Roboflow Serverless API
3. Waste-count and approximate coverage calculation
4. Prototype flood-risk indicator
5. Visual dashboard and CSV export

> **Important:** The flood-risk score is a prototype decision-support
> indicator, not an official hydrological prediction.
"""
    )

    st.markdown("### Prototype Risk Formula")
    st.code(
        "Risk Score = (Detected Waste Count × 8) "
        "+ (Estimated Waste Coverage × 2.5) "
        "+ (Max Confidence × 10)",
        language="text",
    )
