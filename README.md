# 🎯 Real-Time Object Detection & Tracking

A real-time video analytics app built with **YOLOv8 + ByteTrack**, wrapped in a clean **Streamlit** dashboard. Detects and tracks objects in video/webcam streams, with vehicle counting, speed estimation, and live traffic-density prediction.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

## ✨ Features

- **Real-time detection & tracking** — YOLOv8 (Nano/Small/Medium) with persistent object IDs via ByteTrack
- **Vehicle counting** — virtual line-crossing counter with IN/OUT direction
- **Speed estimation** — per-object speed (km/h) from a pixel-to-meter calibration
- **Traffic-density prediction** — rolling Low / Medium / High classification
- **Interactive dashboard** — live metrics, charts, adjustable thresholds, model hot-swap
- **Flexible input** — upload a video file or use your webcam

## 🖥️ Preview

> Add a screenshot or GIF of the running app here, e.g. `assets/demo.gif`

## 📦 Project Structure

```
realtime-object-tracking/
├── app.py                  # Streamlit UI — entry point
├── core/
│   ├── detector.py         # YOLO model loading + tracked inference
│   ├── analytics.py        # Line counter, speed estimator, density meter
│   └── visualizer.py       # OpenCV drawing (boxes, IDs, HUD, line)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/realtime-object-tracking.git
cd realtime-object-tracking
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU users:** install a CUDA-enabled `torch` build first (see [pytorch.org](https://pytorch.org/get-started/locally/)) for much faster inference.

### 2. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### 3. Use it

1. Pick a model in the sidebar (Nano is fastest, Medium is most accurate).
2. Upload a video (or select webcam).
3. Optionally enable vehicle counting / speed estimation and set the pixel calibration.
4. Click **Start Processing** and watch the live annotated feed + metrics.

## ⚙️ Configuration reference

| Setting | What it does |
|---|---|
| Model | Swaps between YOLOv8n / s / m checkpoints |
| Confidence threshold | Minimum detection confidence to keep a box |
| IoU threshold | Non-max-suppression overlap threshold |
| Vehicles only | Restricts detection to car/bus/truck/motorbike (COCO classes) |
| Counting line position | Height (%) at which the virtual counting line sits |
| Pixels per meter | Calibration factor for converting pixel speed to km/h |
| Traffic density smoothing | Number of frames averaged for the Low/Medium/High verdict |
| Frame skip | Process every Nth frame — trade smoothness for speed |

### Calibrating speed estimation

Speed is estimated from pixel displacement, so it needs a scale reference:

1. Find two points in your frame a known real-world distance apart (e.g. a lane width ≈ 3.5 m, or a parked car ≈ 4.5 m long).
2. Measure that distance in pixels (e.g. using any image tool).
3. `pixels_per_meter = pixel_distance / real_distance_in_meters`
4. Enter that value in the sidebar.

Speed readings are an approximation — accuracy depends on camera angle (a top-down or near-perpendicular view works best) and calibration precision.

## 🧠 How it works

1. **Detection** — each frame runs through YOLOv8, producing bounding boxes, class labels, and confidences.
2. **Tracking** — Ultralytics' built-in ByteTrack assigns a persistent ID to each object across frames, enabling counting and speed calculations.
3. **Counting** — a virtual line is checked each frame for a sign-change in which side a track's centroid falls on, indicating a crossing.
4. **Speed** — centroid displacement over a short rolling window is converted from pixels/sec to km/h using the calibration factor.
5. **Density** — a rolling average of vehicles-per-frame is bucketed into Low (≤5), Medium (≤12), or High (>12); thresholds are adjustable in `core/analytics.py`.

## 🛠️ Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection & ByteTrack tracking
- [OpenCV](https://opencv.org/) — video I/O and rendering
- [Streamlit](https://streamlit.io/) — interactive UI
- [Plotly](https://plotly.com/python/) — live charts

## 🗺️ Roadmap / Ideas

- [ ] Export annotated video + CSV of tracked events
- [ ] Multi-line / polygon zone counting
- [ ] Heatmap of traffic density over the frame
- [ ] Swap in RT-DETR / YOLOv9 as alternate backends
- [ ] Dockerfile for one-command deployment

## 🤝 Contributing

Issues and PRs are welcome. Please open an issue first for significant changes so we can discuss the approach.

## 📄 License

Released under the [MIT License](LICENSE).
