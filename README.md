# 電腦視覺作業一：豬隻物件偵測 (TAICA CVPDL 2025 HW1)

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)
![YOLOv10](https://img.shields.io/badge/YOLOv10-Ultralytics-green)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)

This repository contains the solution for the **TAICA CVPDL 2025 HW-1** Kaggle competition. The objective of this homework is to train an object detection model to accurately detect and localize **pigs** within images. 

The project leverages the state-of-the-art **YOLOv10** architecture via the `ultralytics` library, implemented and trained entirely on Google Colab.

## 📁 Repository Structure

```text
.
├── code_M132040009/
│   ├── src/
│   │   └── code.ipynb        # Main Jupyter Notebook containing the end-to-end pipeline
│   ├── README.md             # Original execution steps in Traditional Chinese
│   └── requirements.txt      # Python dependencies
├── report.pdf                # Detailed project report
└── README.md                 # This file
```

## 🚀 Pipeline Overview

The pipeline (`code.ipynb`) is designed to run seamlessly on Google Colab and consists of the following key stages:

1. **Environment Setup & Data Acquisition**
   - Installs necessary dependencies (`ultralytics`, etc.).
   - Authenticates and downloads the dataset directly via Kaggle API.

2. **Data Preprocessing & Splitting**
   - Parses the original ground truth (`gt.txt`) bounding boxes `[x, y, w, h]`.
   - Converts annotations into YOLO normalized format `[class, x_center, y_center, w_norm, h_norm]`.
   - Splits the dataset into **80% Training** and **20% Validation** sets.

3. **Model Training**
   - **Architecture:** YOLOv10 Large (`yolov10l`).
   - **Optimizer:** AdamW.
   - **Hyperparameters:** 100 Epochs, Batch Size 16, Image Size 640.
   - **Advanced Features:** Cosine Learning Rate (`cos_lr`), Automatic Mixed Precision (`amp`), and Early Stopping (`patience=20`).

4. **Inference & Post-processing**
   - Loads the best weights (`best.pt`) from the training phase.
   - Generates predictions on the unseen test set.
   - Converts normalized YOLO predictions back to the required Kaggle submission string format (`conf x_left y_top width height class`).
   - Outputs the final predictions to `submission.csv`.

## 🛠️ Usage / How to Run

1. Open `code_M132040009/src/code.ipynb` in **Google Colab**.
2. Mount your Google Drive.
3. Ensure your `kaggle.json` API token is placed in `/content/drive/MyDrive/.kaggle` to allow the dataset download.
4. Run the notebook cells sequentially from top to bottom.

## 📦 Dependencies

The core libraries used in this project include:
- `ultralytics` (YOLO implementation)
- `opencv-python` (Image processing)
- `pandas` & `numpy` (Data manipulation)
- `albumentations` (Data augmentation)

For a complete list, please refer to the `code_M132040009/requirements.txt` file.
