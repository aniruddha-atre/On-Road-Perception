# scripts/app_streamlit.py
import os
import time
import shutil
import tempfile
import subprocess
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ---------------- UI: header + context ----------------
st.set_page_config(page_title="On-Road Object Detection", layout="wide")
st.title("On-Road Object Detection (YOLO)")

with st.expander("What am I looking at? (Model & Controls)", expanded=False):
    st.markdown("""
**Goal:** Detect common on-road objects (Car/Van/Truck, Pedestrian/Cyclist, Tram, etc.) in images/videos.

**Weights:** Default path points to your fine-tuned YOLO weights on the **KITTI 2D** dataset (7.5k train images).
This app is for **research/demo** only — not for safety-critical use.

**Controls**
- **Device**: `auto` picks GPU if available else CPU. `0` forces GPU 0. `cpu` disables GPU.
- **Confidence**: Minimum score for showing a detection. Higher → fewer boxes (precision↑); lower → more boxes (recall↑).
- **Image size**: Inference resolution. Larger (e.g., 800/960) can improve **mAP@50–95** on small/far objects, but is slower and uses more VRAM.
- **Labels/Confidences**: Toggle text overlays on the visualization.
""")

with st.sidebar:
    st.header("Settings")
    default_weights = r"D:\Projects\on_road_object_detection\on-road-object-detection\y8s_kitti_base2\weights\best.pt"
    weights = st.text_input("Weights path", default_weights)
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.01)
    imgsz = st.select_slider("Image size", options=[480, 640, 736, 800, 960], value=640)
    device = st.selectbox("Device", ["auto", "cpu", "0"])
    show_labels = st.checkbox("Show labels", True)
    show_conf = st.checkbox("Show confidences", True)

@st.cache_resource(show_spinner=False)
def load_model(path, device_choice):
    m = YOLO(path)
    m.to(device=None if device_choice == "auto" else device_choice)
    return m

model = load_model(weights, device)

tab_img, tab_vid = st.tabs(["Image", "Video"])

# ---------------- helpers ----------------
def wait_for_file_ready(path, timeout=12, min_bytes=2048):
    """Wait until file exists, is above min_bytes, and size stops changing."""
    start = time.time()
    last_size = -1
    while time.time() - start < timeout:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size >= min_bytes and size == last_size:
                return True
            last_size = size
        time.sleep(0.2)
    return False

def safe_fps(val, default=30):
    try:
        v = float(val)
        return v if v and v > 0 else default
    except Exception:
        return default

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def transcode_to_h264(in_path: str) -> str | None:
    """
    Make a browser-friendly MP4: H.264 (libx264), yuv420p, faststart.
    Returns output path if successful, else None.
    """
    if not has_ffmpeg():
        return None
    out_path = os.path.splitext(in_path)[0] + "_h264.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        "-an",  # no audio
        out_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if wait_for_file_ready(out_path, timeout=12):
            return out_path
    except Exception:
        pass
    return None

def pretty_results_panel(container, counts: Counter, frames: int, fps: float, imgsz: int, device: str, names: dict):
    total_det = sum(counts.values())
    uniq_cls = len(counts)

    # Top metrics row
    m1, m2, m3, m4 = container.columns(4)
    m1.metric("Frames", f"{frames:,}")
    m2.metric("Throughput", f"{fps:.1f} FPS")
    m3.metric("Detections", f"{total_det:,}")
    m4.metric("Classes", f"{uniq_cls}")

    # Per-class table
    if counts:
        data = [{"Class": names.get(cid, cid), "Count": cnt} for cid, cnt in counts.items()]
        df = pd.DataFrame(data).sort_values("Count", ascending=False, ignore_index=True)
        container.dataframe(df, use_container_width=True, hide_index=True)
    else:
        container.info("No objects detected.")

# ---------------- IMAGE TAB ----------------
with tab_img:
    st.subheader("Run on a single image")
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="image_uploader")
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running inference..."):
            res = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
            vis = res.plot(labels=show_labels, conf=show_conf)

        cls_counts = Counter([int(b.cls.item()) for b in res.boxes]) if res.boxes is not None else Counter()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Predictions", use_container_width=True)
        with col2:
            pretty_results_panel(st, cls_counts, frames=1, fps=0.0, imgsz=imgsz, device=device, names=res.names)

# ---------------- VIDEO TAB ----------------
with tab_vid:
    st.subheader("Run on a video file")
    vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")

    controls_col, results_col = st.columns([1, 2])
    with controls_col:
        run = st.button("Process Video")

    if vid_file is not None and run:
        # clear prior results so we don't show stale/empty players
        results_col.empty()

        # save upload to a temp file (Windows-safe)
        suffix = os.path.splitext(vid_file.name)[1]
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        in_tmp.write(vid_file.read()); in_tmp.flush(); in_tmp.close()

        # We still write mp4 via OpenCV, then transcode to H.264 for browser compatibility
        out_raw = in_tmp.name.replace(suffix, "_out.mp4")
        cap = cv2.VideoCapture(in_tmp.name)
        if not cap.isOpened():
            results_col.error("Could not open the uploaded video.")
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps_in = safe_fps(cap.get(cv2.CAP_PROP_FPS), default=30)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_raw, fourcc, fps_in, (w, h))

            prog = results_col.progress(0.0, text="Processing…")
            preview = results_col.empty()
            panel = results_col.container()

            frame_idx = 0
            cls_counts = Counter()
            t0 = time.time()

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    res = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                    if res.boxes is not None:
                        for b in res.boxes:
                            cls_counts[int(b.cls.item())] += 1

                    vis = res.plot(labels=show_labels, conf=show_conf)
                    writer.write(vis)

                    if frame_idx % 10 == 0:
                        preview.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}", use_container_width=True)

                    frame_idx += 1
                    if total:
                        prog.progress(min(frame_idx / total, 1.0))
            finally:
                cap.release()
                writer.release()

            # ensure raw file is finalized
            if not wait_for_file_ready(out_raw, timeout=12, min_bytes=4096):
                results_col.error("Video not ready for playback (codec/lock). Try re-running or another clip.")
            else:
                # transcode to h264 for browser playback (if possible)
                out_play = transcode_to_h264(out_raw) or out_raw

                dt = max(1e-6, time.time() - t0)
                fps_eff = frame_idx / dt

                results_col.markdown("### Results")
                pretty_results_panel(results_col, cls_counts, frames=frame_idx, fps=fps_eff, imgsz=imgsz, device=device, names=res.names)

                # render ONCE, by file path (h264 if available)
                results_col.video(out_play)

                # download button (use the H.264 file if we made one)
                with open(out_play, "rb") as f:
                    results_col.download_button(
                        "Download annotated video",
                        data=f.read(),
                        file_name="annotated.mp4",
                        mime="video/mp4",
                    )
