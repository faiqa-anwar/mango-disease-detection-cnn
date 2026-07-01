# Mango Leaf Disease Detection

A deep learning project that classifies mango leaf diseases from images, benchmarked against classical computer vision + ML baselines.

## Overview

This project builds a custom deep CNN to classify mango leaf images into 7 disease categories, and compares it against two classical baselines (SVM with HOG features, and Logistic Regression on raw pixels) to evaluate how much deep learning improves over hand-crafted feature approaches.

**Dataset:** [Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset) (Kaggle) — 3,500 images across 7 disease classes (500 each). The "Healthy" class was excluded to keep this a disease-only classification task.

**Classes:** Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould

## Approach

1. **Classical CV preprocessing exploration** — Gaussian blur, Sobel gradients, Canny edge detection, Harris corners, Gabor filters, and CLAHE contrast enhancement, visualized on sample leaves to understand image characteristics.
2. **Deep CNN (main model)** — A compact custom architecture (~103K params) with:
   - 3 conv blocks (Conv2D + BatchNorm + ReLU + MaxPool + Dropout)
   - Heavy data augmentation (flip, rotation, zoom, contrast, brightness, Gaussian noise) to stress-test generalization
   - Global Average Pooling + small dense head with L2 regularization
   - Trained with early stopping and learning-rate reduction on plateau
3. **Classical ML baselines** — for comparison:
   - SVM (RBF kernel) on HOG features
   - Logistic Regression on raw flattened pixels
4. **Ablation study** — a simpler "milestone-faithful" CNN (4.3M params, Flatten instead of GAP, lighter augmentation) trained for direct comparison against the deeper, more heavily regularized model.
5. **Extended evaluation** — ROC/PR curves, top-k accuracy, Grad-CAM visualizations, and misclassification analysis.

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| **Deep CNN** | **0.9257** | 0.9313 | 0.9257 | 0.9261 |
| SVM + HOG | 0.8724 | 0.8732 | 0.8724 | 0.8723 |
| Logistic Regression (pixels) | 0.7486 | 0.7441 | 0.7486 | 0.7453 |

**Additional metrics (Deep CNN):**
- Top-1 accuracy: 92.57%
- Top-2 accuracy: 98.67%
- Top-3 accuracy: 99.43%
- Macro-averaged AUC: 0.9966

### Ablation: model capacity vs. accuracy

| Variant | Test Accuracy | Parameters |
|---|---|---|
| Milestone-faithful CNN (shallow, Flatten) | 0.9638 | 4,296,391 |
| Deep CNN (BN + GAP + L2 + extended aug) | 0.9257 | 102,823 |

The deeper, more heavily regularized model uses ~42x fewer parameters while trading off a few points of accuracy — showing the effect of aggressive augmentation and regularization on a much smaller model.

## Repository Structure

```
├── mango_disease_cnn.ipynb    # Main notebook: data loading, CNN, baselines, ablation, evaluation
├── requirements.txt            # Python dependencies
├── results/
│   ├── summary_metrics.csv         # Model comparison table
│   ├── per_class_metrics.csv       # Precision/recall/F1 per class per model
│   ├── ablation_results.csv        # Ablation study results
│   ├── training_history.json       # Full training curves (loss/accuracy per epoch)
│   └── extended_metrics.json       # ROC/PR/top-k/calibration metrics
└── figures/
    ├── class_distribution.png
    ├── sample_images.png
    ├── cv_preprocessing_pipeline.png
    ├── cnn_curves.png
    ├── confusion_matrices.png
    ├── confusion_matrices_extended.png
    ├── confusion_matrix_normalized.png
    ├── roc_pr_curves.png
    ├── confidence_histogram.png
    ├── gradcam.png
    ├── misclassified.png
    └── ablation.png
```

> **Note:** The trained model file (`best_cnn.keras`) is not included in this repository due to size. Re-run the notebook to reproduce it, or reach out for a copy.

## Setup

```bash
pip install -r requirements.txt
```

Requires a Kaggle API token to download the dataset (the notebook will prompt for it).

## Tech Stack

- **Deep learning:** TensorFlow / Keras
- **Classical ML:** scikit-learn (SVM, Logistic Regression)
- **Feature extraction:** scikit-image (HOG, Gabor filters)
- **Image processing:** OpenCV, Pillow
- **Data & visualization:** NumPy, Pandas, Matplotlib, Seaborn

## Key Techniques Demonstrated

- Custom CNN architecture design with BatchNorm, Dropout, and L2 regularization
- Data augmentation pipeline for small datasets
- Grad-CAM for model interpretability
- Classical CV feature engineering (HOG, Gabor, Canny, Harris corners) as a baseline comparison
- Ablation studies to isolate the effect of architecture choices
- Extended evaluation: ROC/PR curves, top-k accuracy, confidence calibration
