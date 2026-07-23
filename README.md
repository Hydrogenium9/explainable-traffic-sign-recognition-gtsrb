# Explainable Traffic Sign Recognition on GTSRB

This project compares three deep-learning architectures for German
traffic-sign recognition:

- Custom CNN
- EfficientNetV2B0
- ConvNeXtTiny

The project also evaluates robustness under image corruptions and explains
model predictions using Grad-CAM and SHAP.

## Dataset

German Traffic Sign Recognition Benchmark:

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Project stages

1. Data cleaning
2. Group-aware train-validation split
3. Custom CNN training
4. EfficientNetV2B0 transfer learning
5. ConvNeXtTiny transfer learning
6. Model comparison
7. Robustness evaluation
8. Grad-CAM analysis
9. SHAP analysis
10. Streamlit web application

## Repository structure

- `notebooks/` — executed Kaggle notebooks
- `src/` — local Python scripts
- `web_app/` — Streamlit application
- `results/` — compact experiment summaries
- `samples/` — example traffic-sign images
- `models/` — model download instructions

## Installation

```bash
python -m venv .venv