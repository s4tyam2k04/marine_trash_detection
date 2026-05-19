import gradio as gr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
import io
import tempfile
import os
from PIL import Image

REPO_ID = "Krishna-Jaiswal/yolov8m-marine-trash"
model_path = hf_hub_download(repo_id=REPO_ID, filename="yolov8m_marine_best.pt")
model = YOLO(model_path)

CLASS_NAMES = [
    "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
    "animal_crab", "animal_eel", "animal_etc", "trash_etc", "trash_fabric",
    "trash_fishing_gear", "trash_metal", "trash_paper", "trash_plastic",
    "trash_rubber", "trash_wood"
]

TRASH_CLASSES = {"trash_etc","trash_fabric","trash_fishing_gear","trash_metal",
                 "trash_paper","trash_plastic","trash_rubber","trash_wood"}
TRASH_COLOR  = "#D85A30"
MARINE_COLOR = "#1D9E75"
ROV_COLOR    = "#378ADD"

# Class-specific confidence thresholds
# High threshold = overfit classes (dominant in training data, tend to over-predict)
# Low threshold  = underfit classes (rare in training data, need help getting detected)
CLASS_CONF = {
    "rov"               : 0.70,
    "trash_etc"         : 0.60,
    "trash_metal"       : 0.60,
    "animal_fish"       : 0.20,
    "plant"             : 0.25,
    "animal_eel"        : 0.40,
    "animal_starfish"   : 0.25,
    "animal_crab"       : 0.25,
    "animal_etc"        : 0.25,
    "animal_shells"     : 0.25,
    "trash_plastic"     : 0.15,
    "trash_fabric"      : 0.20,
    "trash_paper"       : 0.20,
    "trash_fishing_gear": 0.20,
    "trash_rubber"      : 0.15,
    "trash_wood"        : 0.20,
}

def filter_detections(result, base_conf):
    """Apply class-specific confidence thresholds."""
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_name = CLASS_NAMES[int(box.cls[0])]
            conf     = float(box.conf[0])
            threshold = CLASS_CONF.get(cls_name, base_conf)
            if conf >= threshold:
                detections.append((cls_name, conf))
    return detections

def build_dashboard(detections):
    if not detections:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No detections above threshold",
                ha="center", va="center", fontsize=13,
                color="#7a9bbf", transform=ax.transAxes)
        ax.axis("off")
        fig.patch.set_facecolor("#0d1b2a")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1b2a")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    names  = [d[0] for d in detections]
    confs  = [d[1] for d in detections]
    counts = Counter(names)

    fig = plt.figure(figsize=(12, 4), facecolor="#0d1b2a")
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.97, top=0.82, bottom=0.20)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor("#111e2e")
    classes = list(counts.keys())
    values  = list(counts.values())
    colors  = [TRASH_COLOR if c in TRASH_CLASSES else (ROV_COLOR if c == "rov" else MARINE_COLOR) for c in classes]
    bars = ax1.barh(classes, values, color=colors, height=0.55, edgecolor="none")
    ax1.bar_label(bars, padding=4, color="#cdd9e5", fontsize=9)
    ax1.set_xlabel("Count", color="#7a9bbf", fontsize=9)
    ax1.set_title("Detected classes", color="#cdd9e5", fontsize=10, pad=8)
    ax1.tick_params(colors="#7a9bbf", labelsize=8)
    for s in ax1.spines.values(): s.set_edgecolor("#1e3450")
    ax1.set_xlim(0, max(values) + 2)
    ax1.legend(handles=[
        mpatches.Patch(color=TRASH_COLOR,  label="Trash"),
        mpatches.Patch(color=MARINE_COLOR, label="Marine life"),
        mpatches.Patch(color=ROV_COLOR,    label="ROV"),
    ], fontsize=8, facecolor="#111e2e", edgecolor="#1e3450", labelcolor="#cdd9e5", loc="lower right")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#111e2e")
    conf_colors = [TRASH_COLOR if n in TRASH_CLASSES else (ROV_COLOR if n == "rov" else MARINE_COLOR) for n in names]
    ax2.bar(range(len(names)), confs, color=conf_colors, width=0.6, edgecolor="none")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace("animal_","").replace("trash_","") for n in names],
                        rotation=35, ha="right", fontsize=8, color="#7a9bbf")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Confidence", color="#7a9bbf", fontsize=9)
    ax2.set_title("Confidence scores", color="#cdd9e5", fontsize=10, pad=8)
    ax2.tick_params(colors="#7a9bbf", labelsize=8)
    ax2.axhline(0.5, color="#ffffff22", linewidth=0.8, linestyle="--")
    for s in ax2.spines.values(): s.set_edgecolor("#1e3450")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1b2a")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def make_summary(detections):
    if not detections:
        return "### No objects detected\nTry lowering the confidence threshold."
    trash_count  = sum(1 for d in detections if d[0] in TRASH_CLASSES)
    marine_count = sum(1 for d in detections if d[0] not in TRASH_CLASSES and d[0] != "rov")
    rov_count    = sum(1 for d in detections if d[0] == "rov")
    total        = len(detections)
    lines = [
        f"### {total} object(s) detected\n",
        "| Category | Count |", "|---|---|",
        f"| Trash | {trash_count} |",
        f"| Marine life | {marine_count} |",
        f"| ROV | {rov_count} |",
        "\n**Detected objects:**",
    ]
    for cls_name, conf in detections:
        cat = "Trash" if cls_name in TRASH_CLASSES else ("ROV" if cls_name == "rov" else "Marine life")
        lines.append(f"- `{cls_name}` ({cat}) — {conf:.0%}")
    return "\n".join(lines)
def predict_image(image):
    if image is None:
        return None, None, "Upload an image to begin."
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    result = model.predict(source=image, conf=0.15, task="segment", verbose=False, iou=0.5)[0]
    detections = filter_detections(result, 0.25)
    kept_indices = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            cls_name  = CLASS_NAMES[int(box.cls[0])]
            c         = float(box.conf[0])
            threshold = CLASS_CONF.get(cls_name, 0.25)
            if c >= threshold:
                kept_indices.append(i)
    if kept_indices and result.boxes is not None:
        result.boxes = result.boxes[kept_indices]
        if result.masks is not None:
            result.masks = result.masks[kept_indices]
    annotated = cv2.cvtColor(
        result.plot(masks=True, boxes=True, conf=True, labels=True),
        cv2.COLOR_BGR2RGB
    )
    return annotated, build_dashboard(detections), make_summary(detections)

def predict_video(video_path):
    if video_path is None:
        return None, None, "Upload a video to begin."
    if isinstance(video_path, dict):
        video_path = video_path.get("video", video_path.get("name", ""))
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    MAX_FRAMES   = 300
    out_path     = tempfile.mktemp(suffix=".mp4")
    writer       = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    all_detections = []
    frame_counts   = defaultdict(int)
    frame_idx      = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= MAX_FRAMES:
            break
        result = model.predict(source=frame, conf=0.15, task="segment", verbose=False, iou=0.5)[0]
        kept_indices = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls_name  = CLASS_NAMES[int(box.cls[0])]
                c         = float(box.conf[0])
                if c >= CLASS_CONF.get(cls_name, 0.25):
                    kept_indices.append(i)
                    all_detections.append((cls_name, c))
                    frame_counts[cls_name] += 1
        if kept_indices and result.boxes is not None:
            result.boxes = result.boxes[kept_indices]
            if result.masks is not None:
                result.masks = result.masks[kept_indices]
        writer.write(result.plot(masks=True, boxes=True, conf=True, labels=True))
        frame_idx += 1
    cap.release()
    writer.release()
    dashboard = build_dashboard(all_detections)
    if not all_detections:
        summary = "### No objects detected\nTry lowering the confidence threshold."
    else:
        trash_total  = sum(1 for d in all_detections if d[0] in TRASH_CLASSES)
        marine_total = sum(1 for d in all_detections if d[0] not in TRASH_CLASSES and d[0] != "rov")
        rov_total    = sum(1 for d in all_detections if d[0] == "rov")
        lines = [
            f"### Video analysis — {min(frame_idx, MAX_FRAMES)} frames processed\n",
            "| Category | Total detections |", "|---|---|",
            f"| Trash | {trash_total} |",
            f"| Marine life | {marine_total} |",
            f"| ROV | {rov_total} |",
            "\n**Most detected classes:**",
        ]
        for cls_name, count in sorted(frame_counts.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- `{cls_name}` — {count} detections across frames")
        if total_frames > MAX_FRAMES:
            lines.append(f"\n*Note: processed first {MAX_FRAMES} frames of {total_frames} total.*")
        summary = "\n".join(lines)
    return out_path, dashboard, summary

CSS = """
body { background: #0d1b2a !important; }
.gradio-container { background: #0d1b2a !important; font-family: 'Segoe UI', sans-serif; }
h1, h2, h3 { color: #cdd9e5 !important; }
p, label { color: #7a9bbf !important; }
.gr-button-primary { background: #065A82 !important; border: none !important; color: white !important; font-weight: 500 !important; }
.gr-button-primary:hover { background: #1C7293 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Life Under Water") as demo:
    gr.HTML("""
    <div style="padding: 1.5rem 0 0.5rem;">
        <h1 style="font-size: 2rem; font-weight: 700; color: #378ADD; margin: 0;">Life Under Water</h1>
        <p style="font-size: 1rem; color: #7a9bbf; margin: 4px 0 0;">Marine Trash Detection System</p>
    </div>
    """)
    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload underwater image", type="pil", height=360)
                    img_btn   = gr.Button("Run Detection", variant="primary")
                with gr.Column():
                    img_output = gr.Image(label="Segmentation output", height=360)
            gr.Examples(
                examples=[
                    ["examples/sample1.jpg"],
                    ["examples/sample2.jpg"],
                ],
                inputs=img_input,
                label="Try a sample",
                examples_per_page=3,
            )
            img_dashboard = gr.Image(label="Detection stats dashboard", height=280)
            img_summary   = gr.Markdown()
            img_btn.click(fn=predict_image, inputs=[img_input],
                          outputs=[img_output, img_dashboard, img_summary])
        with gr.Tab("Video"):
            gr.Markdown("Upload an underwater video. The model processes it frame by frame and returns annotated output with segmentation masks.")
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload underwater video")
                    vid_btn   = gr.Button("Analyse Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Annotated output video")
            gr.Examples(
                examples=[["examples/sample_video.mp4"]],
                inputs=vid_input,
                label="Try a sample",
            )
            vid_dashboard = gr.Image(label="Aggregated stats dashboard (all frames)", height=280)
            vid_summary   = gr.Markdown()
            vid_btn.click(fn=predict_video, inputs=[vid_input],
                          outputs=[vid_output, vid_dashboard, vid_summary])


if __name__ == "__main__":
    demo.launch()
