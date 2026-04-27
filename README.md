# 🌿 WeedSense

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat&logo=flask&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.38%25-brightgreen?style=flat)

**AI-powered weed detection system for precision agriculture** — Binary classification of soybean crops vs. weeds using deep learning.


## 🎯 Overview

WeedSense is a web-based weed detection application that uses a Convolutional Neural Network (CNN) to classify agricultural field images as either **Soybean (Crop)** or **Weed/Other**. Trained on 15,336 images with **99.38% validation accuracy**, it helps farmers identify weed presence for targeted herbicide application.

### ✨ Key Features

- 🚀 **High Accuracy** — 99.38% validation accuracy on real-world agricultural data
- 🖼️ **Multi-Format Support** — JPG, PNG, and TIF image formats
- 📋 **Clipboard Paste** — Ctrl+V to paste images directly from clipboard
- 📊 **Real-Time Analysis** — Live progress bar with percentage updates
- 📱 **Fully Responsive** — Works seamlessly on desktop, tablet, and mobile
- 🎨 **Modern UI** — Clean dark theme with intuitive interface
- ⚡ **Fast Inference** — Optimized CNN architecture for quick predictions

 
## 🏗️ Architecture

### Model Details

- **Type:** Convolutional Neural Network (CNN)
- **Task:** Binary Classification
- **Input:** 128×128×3 RGB images
- **Output:** Soybean (1) or Weed/Other (0)
- **Parameters:** 3.3M trainable parameters
- **Framework:** TensorFlow 2.21

### Network Structure

```
Input (128×128×3)
    ↓
Conv2D (32 filters, 3×3) + ReLU → MaxPool (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU → MaxPool (2×2)
    ↓
Conv2D (128 filters, 3×3) + ReLU → MaxPool (2×2)
    ↓
Flatten (25,088) → Dense (128) + ReLU → Dropout (0.3)
    ↓
Dense (1) + Sigmoid
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | Weed Detection in Soybean Crops (Kaggle) |
| **Total Images** | 15,336 (7,376 soybean, 3,520 grass, 1,191 broadleaf, 3,249 soil) |
| **Train/Val Split** | 80% / 20% (12,268 / 3,068) |
| **Batch Size** | 32 |
| **Optimizer** | Adam (LR: 0.001 → 0.00025) |
| **Loss Function** | Binary Crossentropy |
| **Augmentation** | Flip LR/UD, Random Brightness |
| **Callbacks** | EarlyStopping (patience=4), ReduceLROnPlateau (patience=2) |
| **Best Epoch** | 11 / 15 |
| **Val Accuracy** | **99.38%** |
| **Val Loss** | 0.0142 |


## 📊 Dataset

**Source:** [Kaggle — Weed Detection in Soybean Crops](https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops) by fpeccia

### Class Distribution

| Class | Count | Label |
|-------|-------|-------|
| Soybean | 7,376 | Crop (1) |
| Grass | 3,520 | Weed (0) |
| Broadleaf | 1,191 | Weed (0) |
| Soil | 3,249 | Other (0) |

All images are in `.tif` format and represent real agricultural field conditions.
---

<div align="center">
  <strong>Made with ❤️ for precision agriculture</strong>
</div>
