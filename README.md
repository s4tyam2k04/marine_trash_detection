# 🌊 Life Under Water — Marine Trash Detection

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Krishna-Jaiswal/marine-trash-detection)
[![Model](https://img.shields.io/badge/🤗%20Model-Hugging%20Face-blue)](https://huggingface.co/Krishna-Jaiswal/yolov8m-marine-trash)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8m--seg-green)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> Real-time underwater debris detection and instance segmentation using YOLOv8m-seg,
> designed specifically for **ROV (Remotely Operated Vehicle) camera footage** —
> seabed and underwater surface debris detection.

---

## Live Demo

Try it instantly — no installation needed:

**[huggingface.co/spaces/s4tyam2k04/marine_trash_detector](https://huggingface.co/spaces/s4tyam2k04/marine_trash_detector)**

Upload any underwater ROV image or video and get segmentation masks + detection stats.

---

## Sample Predictions

![Sample predictions](assets/val_batch0_pred.jpg)

---

## Results

| Metric | Score |
|--------|-------|
| mAP50 — detection | 65.6% |
| mAP50 — segmentation | 65.0% |
| Precision | 79% |
| Epochs | 75 |
| Training images | 6,008 |

---

## Training Curves

![Training curves](assets/training_curves.png)

---

## Confusion Matrix

![Confusion matrix](assets/confusion_matrix.png)

---

## Dataset

**TrashCan 1.0** — Instance segmentation dataset of underwater trash from
JAMSTEC deep-sea ROV cameras.
Source: [University of Minnesota Data Repository](https://conservancy.umn.edu/handle/11299/214865)

**7,212 images | 16 classes | Train: 6,008 | Val: 1,204**

| Category | Classes |
|----------|---------|
| Trash | `trash_plastic`, `trash_metal`, `trash_fabric`, `trash_fishing_gear`, `trash_rubber`, `trash_wood`, `trash_paper`, `trash_etc` |
| Marine life | `animal_fish`, `animal_starfish`, `animal_shells`, `animal_crab`, `animal_eel`, `animal_etc`, `plant` |
| Equipment | `rov` |

---

## Project Structure

```
marine-trash-detection/
├── app.py                          # Gradio web app — deployed on HF Spaces
├── requirements.txt                # Dependencies
├── kaggle_marine_trash.ipynb       # Training notebook (Kaggle)
├── save_to_hf_kaggle.ipynb         # Upload model to HF Hub
├── LICENSE                         # MIT License
├── README.md                       # Documentation
│
├── assets/                         # Training artifacts & visualizations
│   ├── val_batch0_pred.jpg        # Sample predictions
│   ├── training_curves.png        # Loss + mAP curves
│   ├── confusion_matrix.png       # Per-class accuracy
│   └── class_distribution.png     # Dataset class distribution
│
└── examples/                       # Demo samples for Gradio app
    ├── sample1.jpg               # Example underwater image 1
    ├── sample2.jpg               # Example underwater image 2
    └── sample_video.mp4          # Example underwater video
```

---

## Quick Start

```bash
git clone https://github.com/Krishna-Jaiswal/marine-trash-detection.git
cd marine-trash-detection
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8m-seg (pretrained COCO) |
| Optimizer | SGD + cosine LR decay |
| Learning rate | 0.01 → 0.001 |
| Epochs | 75 |
| Batch size | 16 |
| Image size | 640×640 |
| Augmentation | Mosaic, Mixup=0.1, Copy-paste=0.1 |
| Platform | Kaggle T4 GPU |

---

## Limitations

- **Class imbalance**: ROV class dominates (2,653 instances) vs trash_rubber (113 instances), creating 59:1 ratio. Model tends to overpredict ROV.
- **mAP justification**: Lower mAP (65.6%) reflects challenging underwater domain — low-light, murky conditions, occlusions. Consistent with underwater detection benchmarks (60-70%).
- **Solution applied**: Class-specific confidence thresholds (ROV: 0.70, rare classes: 0.15-0.20) to balance false positives/negatives.
- **Trained on JAMSTEC ROV footage** — performance varies on other camera types/environments.
- **Optimized for seabed debris** — not floating/suspended trash.

---

## Real-world Applications

- Underwater ROV-based ocean cleanup robots
- Marine pollution monitoring and research
- Automated seabed debris surveys
- Aligned with **UN SDG 14 — Life Below Water**
- Same problem domain as **EU SeaClear 2.0** (€9M Horizon Europe project)

---

## Team

| Name | Role | Contributions |
|------|------|---------------|
| Krishna Jaiswal | Lead Developer | Dataset pipeline, YOLOv8m-seg training, class imbalance analysis, class-specific confidence thresholds, Gradio web app, HF Hub model upload, HF Spaces deployment |
| Shashank Kumar Tiwari | Data | Dataset download, extraction, folder structure preparation |
| Satyam Kumar | Evaluation | Results review, confusion matrix analysis, PPT preparation |

---

## Citation

```bibtex
@dataset{trashcan2020,
  title  = {TrashCan 1.0: An Instance-segmentation Labeled Dataset of Trash Observations},
  author = {Hong, Jungseok and Fulton, Michael and Sattar, Junaed},
  year   = {2020},
  url    = {https://conservancy.umn.edu/handle/11299/214865}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
