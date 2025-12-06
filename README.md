# 🐾 Animal Image Classifier

An experimental study comparing different **feature extraction methods** (SIFT, HOG, VGG16) combined with **machine learning classifiers** (SVM, Random Forest) for animal image classification.

## 📋 Overview

This project is part of the **CS231 - Computer Vision** course, exploring and comparing various feature extraction techniques for image classification:
## 👥 Team Members

| Name | Student ID | Role |
|------|------------|------|
| **Lương Quang Duy** | 23520368 | 
| **Trần Minh Nhất** | 23521101 |
| **Dương Thái Ý Nhi** |23521106 | 


| Method | Feature Type |
|--------|--------------|
| **SIFT** | Handcrafted (keypoints)|
| **HOG** | Handcrafted (gradients)|
| **VGG16** | Deep learning (CNN)|

Each method is evaluated with two classifiers:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**

> **Best Result**: VGG16 + SVM achieved the highest accuracy and is used in the demo application.

### Supported Animal Classes
`butterfly` | `cat` | `chicken` | `cow` | `dog` | `elephant` | `horse` | `sheep` | `spider` | `squirrel`

## 🏗️ VGG16 Pipeline (Best Method)

```
Input Image (any size)
        ↓
Preprocessing (resize to 224×224, RGB conversion)
        ↓
VGG16 Feature Extraction (fc2 layer → 4096-dim vector)
        ↓
StandardScaler (normalize features)
        ↓
PCA (reduce to 500 components)
        ↓
Classifier (SVM or Random Forest)
        ↓
Predicted Animal Class
```

## 📁 Project Structure

```
proj/
├── app.py                      # Gradio web UI (VGG16 - best method)
├── main.py                     # Entry point
├── pyproject.toml              # Project dependencies
├── README.md
├── demo_imgs/                  # Sample images for testing
├── modules/                    # Pre-trained model components (VGG16)
│   ├── vgg16_label_encoder.pkl
│   ├── vgg16_scaler.pkl
│   ├── vgg16_pca.pkl
│   ├── vgg16_svm_model.pkl
│   └── vgg16_random_forest_model.pkl
├── notebooks/
│   ├── CS231_SIFT.ipynb        # SIFT experiment (coming soon)
│   ├── CS231_HOG.ipynb         # HOG experiment (coming soon)
│   ├── CS231_VGG16.ipynb       # VGG16 experiment
│   └── demo_prediction.ipynb   # Demo notebook
└── src/
    └── predict.py              # Classifier class and CLI
```

## 🚀 Getting Started

### Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd proj

# Install dependencies
uv sync
```

### Usage

#### 1. Web UI (Gradio) - Recommended

```bash
uv run app.py
```

Open `http://localhost:7860` in your browser.

Features:
- **Single Prediction**: Upload an image and select SVM or Random Forest
- **Compare Models**: See predictions from both models side-by-side

#### 2. Command Line

```bash
# Predict with SVM (default)
python src/predict.py path/to/image.jpg --model svm

# Predict with Random Forest
python src/predict.py path/to/image.jpg --model random_forest

# Interactive mode
python src/predict.py
```

#### 3. Jupyter Notebook

```bash
# Run experiments
jupyter notebook notebooks/CS231_VGG16.ipynb

# Demo predictions
jupyter notebook notebooks/demo_prediction.ipynb
```

#### 4. Python API

```python
from src.predict import AnimalClassifier

# Initialize classifier
classifier = AnimalClassifier('modules', model_type='svm')

# Make prediction
result = classifier.predict('path/to/image.jpg')

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🔧 Dependencies

| Package | Version |
|---------|---------|
| tensorflow-cpu | >=2.20.0 |
| scikit-learn | ==1.6.1 |
| opencv-python | >=4.11.0 |
| numpy | >=2.3.5 |
| gradio | >=6.0.2 |
| matplotlib | >=3.10.7 |

## 📝 Notes

- First prediction may take longer as models are loaded into memory
- Uses `tensorflow-cpu` by default; for GPU support, replace with `tensorflow`
- scikit-learn version is pinned to 1.6.1 to match the saved model files

## 📚 Course Information

**Course**: CS231 - Computer Vision  
**University**: University of Information Technology (UIT)

## 🙏 Acknowledgments

- [VGG16](https://arxiv.org/abs/1409.1556) - Visual Geometry Group, Oxford
- [Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) - Kaggle
- [Gradio](https://gradio.app/) - Web UI framework
