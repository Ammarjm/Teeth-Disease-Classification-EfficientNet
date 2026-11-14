<div align="center">

# 🦷 Teeth Disease Classification — EfficientNetB0

**Transfer learning with EfficientNetB0 to classify dental images into 7 conditions, deployed as an interactive Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNetB0-4285F4)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97%25-success)

</div>

---

## 📌 Overview

This project compares two approaches to dental image classification:

1. **Custom CNN** trained from scratch — see [Teeth-Classification-CNN](https://github.com/Ammarjm/Teeth-Classification-CNN)
2. **EfficientNetB0** pre-trained on ImageNet, adapted with **transfer learning and fine-tuning** (this repository)

The final model is served through a **Streamlit** web app: upload a dental image and get the predicted condition instantly.

## 🏷️ Classes

`CaS` Caries · `CoS` Calculus · `Gum` Gum Disease · `MC` Mouth Cancer · `OC` Oral Cancer · `OLP` Oral Lichen Planus · `OT` Other

**Dataset split:** 3,087 training · 1,028 validation · 1,028 test images

## ⚙️ Training Strategy

```mermaid
flowchart LR
    A[Dental Images] --> B[Preprocess & Augment]
    B --> C[EfficientNetB0 · frozen]
    C --> D[Train custom head]
    D --> E[Unfreeze & fine-tune]
    E --> F[Evaluate on test set]
    F --> G[Streamlit App]
```

- **Stage 1 — Feature extraction:** frozen EfficientNetB0 backbone with a new classification head (≈ 330 K trainable parameters)
- **Stage 2 — Fine-tuning:** backbone unfrozen and trained with a low learning rate
- **Balanced class weights** to handle uneven class sizes
- **Callbacks:** early stopping · learning-rate reduction · best-model checkpointing

## 📊 Results (Test Set)

| Metric | Score |
|---|---|
| Accuracy | **97 %** |
| Precision | 0.97 |
| Recall | 0.97 |
| F1-Score | **0.97** |

## 🧰 Tech Stack

Python · TensorFlow / Keras · EfficientNetB0 · NumPy · Pandas · OpenCV · Scikit-learn · Matplotlib · Seaborn · Streamlit

## 📂 Project Structure

```text
├── Transfer_Learning.ipynb   # Training, fine-tuning & evaluation
├── app.py                    # Streamlit app
├── models/                   # Saved models (head weights & fine-tuned model)
├── Teeth_Dataset/            # Training / Validation / Testing images
└── requirements.txt
```

## 🚀 Getting Started

```bash
git clone https://github.com/Ammarjm/Teeth-Disease-Classification-EfficientNet.git
cd Teeth-Disease-Classification-EfficientNet
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author :

**Ammar Alsabateen** — AI & Machine Learning Engineer
[GitHub](https://github.com/Ammarjm) · [Email](mailto:alsabateenammar@gmail.com)
