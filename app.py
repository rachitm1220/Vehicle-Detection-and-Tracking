"""
app.py
------
Streamlit front-end for the Real-Time Object Detection & Tracking system.

Run with:
    streamlit run app.py
"""

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.analytics import LineCounter, SpeedEstimator, TrafficDensityMeter
from core.detector import ObjectTracker
from core.visualizer import draw_counting_line, draw_detections, draw_hud

# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #9ca3af; font-size: 0.95rem; margin-bottom: 1.4rem; }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #262b36;
        padding: 14px 16px;
        border-radius: 12px;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
    .badge-low { color: #22c55e; font-weight: 700; }
    .badge-medium { color: #f59e0b; font-weight: 700; }
    .badge-high { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Real-Time Object Detection & Tracking</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">YOLOv8 + ByteTrack · vehicle counting · speed estimation · traffic-density prediction</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "tracker" not in st.session_state:
    st.session_state.tracker = None
if "model_key" not in st.session_state:
    st.session_state.model_key = None
if "history" not in st.session_state:
    st.session_state.history = []  # per-frame stats for charts

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    model_key = st.selectbox("Model", list(ObjectTracker.AVAILABLE_MODELS.keys()), index=0)
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    iou_thresh = st.slider("IoU (NMS) threshold", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.subheader("🚗 Vehicle features")
    vehicle_only = st.checkbox("Vehicles only (car/bus/truck/motorbike)", value=True)
    enable_counting = st.checkbox("Enable line-crossing count", value=True)
    line_y_pct = st.slider("Counting line position (% height)", 10, 90, 60, disabled=not enable_counting)

    enable_speed = st.checkbox("Enable speed estimation", value=True)
    pixels_per_meter = st.number_input(
        "Calibration: pixels per meter", min_value=1.0, value=8.0, step=0.5,
        help="Estimate this from a known real-world distance visible in your video "
             "(e.g. lane width ≈ 3.5 m). Roughly: (pixel length of that distance) / (meters).",
        disabled=not enable_speed,
    )

    st.divider()
    density_window = st.slider("Traffic density smoothing (frames)", 5, 90, 30)

    st.divider()
    frame_skip = st.slider("Process every Nth frame (speed vs. smoothness)", 1, 5, 1)

    load_btn = st.button("🔄 Load / Reload Model", width='stretch')

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorbike, bus, truck

if load_btn or st.session_state.tracker is None or st.session_state.model_key != model_key:
    with st.spinner(f"Loading {model_key}..."):
        st.session_state.tracker = ObjectTracker(model_key)
        st.session_state.model_key = model_key
    st.sidebar.success("Model ready ✅")

# --------------------------------------------------------------------------
# Main layout: input source
# --------------------------------------------------------------------------
tab_upload, tab_about = st.tabs(["📹 Run Detection", "ℹ️ About"])

with tab_upload:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        source = st.radio("Video source", ["Upload a video", "Use webcam"], horizontal=True)
        uploaded_file = None
        if source == "Upload a video":
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        run_btn = st.button("▶️ Start Processing", type="primary", width='stretch')
        stop_btn_placeholder = st.empty()
        video_placeholder = st.empty()

    with col_right:
        st.subheader("📊 Live Stats")
        metric_objects = st.empty()
        metric_fps = st.empty()
        metric_density = st.empty()
        metric_counts = st.empty()
        chart_placeholder = st.empty()

    if run_btn:
        cap = None
        if source == "Upload a video" and uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        elif source == "Use webcam":
            cap = cv2.VideoCapture(0)
        else:
            st.warning("Please upload a video file first.")

        if cap is not None and cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            line = ((0, int(height * line_y_pct / 100)), (width, int(height * line_y_pct / 100)))
            line_counter = LineCounter(line) if enable_counting else None
            speed_estimator = SpeedEstimator(pixels_per_meter) if enable_speed else None
            density_meter = TrafficDensityMeter(window=density_window)

            classes_filter = VEHICLE_CLASS_IDS if vehicle_only else None

            st.session_state.history = []
            stop = stop_btn_placeholder.button("⏹ Stop", width='stretch')
            frame_idx = 0
            prev_t = time.time()

            while cap.isOpened() and not stop:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                detections = st.session_state.tracker.track(
                    frame, conf=conf_thresh, iou=iou_thresh, classes=classes_filter
                )

                if speed_estimator:
                    speed_estimator.update(detections)
                    speed_estimator.cleanup([d.track_id for d in detections if d.track_id is not None])

                if line_counter:
                    line_counter.update(detections)

                vehicle_count = sum(1 for d in detections if d.class_id in VEHICLE_CLASS_IDS)
                density = density_meter.update(vehicle_count)

                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now

                annotated = frame.copy()
                speeds = speed_estimator.speeds_kmh if speed_estimator else {}
                annotated = draw_detections(annotated, detections, speeds)
                if line_counter:
                    annotated = draw_counting_line(annotated, line, line_counter.count_in, line_counter.count_out)
                annotated = draw_hud(annotated, fps, density, len(detections))

                video_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width='stretch',
                )

                metric_objects.metric("Objects in frame", len(detections))
                metric_fps.metric("Processing FPS", f"{fps:.1f}")
                badge_class = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[density]
                metric_density.markdown(f"**Traffic density:** <span class='{badge_class}'>{density}</span>",
                                         unsafe_allow_html=True)
                if line_counter:
                    metric_counts.metric("Line crossings (total)", line_counter.total,
                                          f"IN {line_counter.count_in} / OUT {line_counter.count_out}")

                st.session_state.history.append({
                    "frame": frame_idx, "objects": len(detections),
                    "vehicles": vehicle_count, "fps": fps,
                    "avg_speed": (sum(speeds.values()) / len(speeds)) if speeds else 0,
                })

                if len(st.session_state.history) % 10 == 0:
                    df = pd.DataFrame(st.session_state.history)
                    fig = px.line(df, x="frame", y=["objects", "vehicles"],
                                   title="Detections over time", template="plotly_dark")
                    chart_placeholder.plotly_chart(fig, width='stretch', key=f"chart_{frame_idx}")

            cap.release()
            st.success("Processing finished." if not stop else "Stopped by user.")

with tab_about:
    st.markdown("""
### How it works
1. **Detection** — each frame is run through a YOLOv8 model (Nano/Small/Medium).
2. **Tracking** — Ultralytics' built-in **ByteTrack** assigns a persistent ID to every object across frames.
3. **Vehicle counting** — a virtual line is drawn across the frame; when a tracked centroid crosses it, the count increments (IN vs OUT based on direction).
4. **Speed estimation** — pixel displacement of each tracked centroid, converted to real-world speed using a **pixels-per-meter calibration** you provide.
5. **Traffic density** — a rolling average of vehicles-per-frame is bucketed into **Low / Medium / High**.

### Tips for accurate speed estimates
Pick two points a known real-world distance apart (e.g. lane markings, a car length) and divide
their pixel distance by the real distance in meters to get `pixels_per_meter`.

### Model choice
| Model | Speed | Accuracy | Best for |
|---|---|---|---|
| YOLOv8 Nano | ⚡⚡⚡ | ★★ | CPU / webcam / low-power |
| YOLOv8 Small | ⚡⚡ | ★★★ | Balanced default |
| YOLOv8 Medium | ⚡ | ★★★★ | GPU / offline accuracy |
""")
