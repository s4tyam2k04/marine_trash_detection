````md
# 🌊 Life Under Water — Marine Trash Detection

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/s4tyam2k04/marine_trash_detector)
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

```text
marine-trash-detection/
├── app.py                          # Gradio web app deployed on HF Spaces
├── requirements.txt                # Dependencies
├── kaggle_marine_trash.ipynb       # Training notebook
├── save_to_hf_kaggle.ipynb         # Upload model to HF Hub
├── LICENSE                         # MIT License
├── README.md                       # Documentation
│
├── assets/
│   ├── val_batch0_pred.jpg
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── class_distribution.png
│
└── examples/
    ├── sample1.jpg
    ├── sample2.jpg
    └── sample_video.mp4
````

---

## Quick Start

```bash
git clone https://github.com/s4tyam2k04/marine-trash-detection.git
cd marine-trash-detection

pip install -r requirements.txt

python app.py
```

Open:

```text
http://localhost:7860
```

---

## Training Configuration

| Parameter     | Value                             |
| ------------- | --------------------------------- |
| Model         | YOLOv8m-seg (pretrained COCO)     |
| Optimizer     | SGD + cosine LR decay             |
| Learning rate | 0.01 → 0.001                      |
| Epochs        | 75                                |
| Batch size    | 16                                |
| Image size    | 640×640                           |
| Augmentation  | Mosaic, Mixup=0.1, Copy-paste=0.1 |
| Platform      | Kaggle T4 GPU                     |

---

## Limitations

* **Class imbalance:** ROV class dominates (2,653 instances) vs trash_rubber (113 instances), creating a large class imbalance.
* **mAP justification:** Underwater environments contain low-light conditions, murky water, motion blur, and occlusions that reduce detection accuracy.
* **Solution applied:** Class-specific confidence thresholds (ROV: 0.70, rare classes: 0.15–0.20) to balance false positives and false negatives.
* **Trained on JAMSTEC ROV footage** — performance may vary on different underwater environments and camera systems.
* **Optimized for seabed debris** rather than floating or suspended waste.

---

## Real-world Applications

* Underwater ROV-based ocean cleanup robots
* Marine pollution monitoring and research
* Automated seabed debris surveys
* Supports **UN SDG 14 — Life Below Water**
* Environmental conservation and sustainability initiatives

---

## Team

| Name                  | Role                        | Contributions                                                                                                                                           |
| --------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Satyam Kumar**      | **Deployment & Evaluation** | Hugging Face Spaces deployment, model testing, evaluation, confusion matrix analysis, documentation, PPT preparation                                    |
| Krishna Jaiswal       | Lead Developer              | Dataset pipeline, YOLOv8m-seg training, class imbalance analysis, class-specific confidence thresholds, Gradio web app development, HF model deployment |
| Shashank Kumar Tiwari | Data                        | Dataset download, extraction, preprocessing, folder structure preparation                                                                               |

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

```
```
