# LoRA-Based-Intrusion-Detection-for-Lightweight-Security

## Overview

This project implements a **resource-efficient Intrusion Detection System (IDS)** using **Low-Rank Adaptation (LoRA)** fine-tuned Large Language Models (LLMs) and traditional machine learning approaches.
The system is designed to detect **cyberattacks** in IoT or network environments while maintaining **low computational cost** and **high accuracy** — making it suitable for **lightweight edge security applications**.

---

## Objective

To develop a **lightweight, high-performance intrusion detection framework** capable of classifying network traffic as **benign or attack** using fine-tuned models and optimized architectures.

---

## Key Features

* 🔹 **Multi-model comparison:** LightGBM, Random Forest, and 1D-CNN
* 🔹 **Low resource usage** suitable for edge and IoT devices
* 🔹 **Streamlit-based frontend** for easy model evaluation and comparison
* 🔹 **Preprocessing pipeline** for consistent feature scaling and input normalization

---

## Models Implemented

| Model             | Description                                                                       |
| ----------------- | --------------------------------------------------------------------------------- |
| **LightGBM**      | Gradient boosting model optimized for speed and accuracy on structured data.      |
| **Random Forest** | Ensemble model providing strong baseline performance and interpretability.        |
| **1D-CNN**        | Deep learning model that captures complex sequential patterns in network traffic. |

---

## Project Structure

```
project_ids_/
│
├── intrusion_detection/
│   ├── cnn_train.py
│   ├── cnn_infer.py
│   ├── rf_train.py
│   ├── rf_infer.py
│   ├── lightgbm_train.py
│   ├── lightgbm_infer.py
│   ├── preprocess.py
│   ├── cnn_model.pt
│   ├── rf_model.joblib
│   ├── lgb_model.txt
│   ├── *_scaler.pkl
│   └── test_data.pkl
│
├── frontend.py        # Streamlit app for model comparison
├── venv/              # Virtual environment (not pushed to Git)
└── data/              # Dataset folder
```

---

## How to Run

### Setup Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Train Models (Optional)

Run the training scripts in the `intrusion_detection/` folder:

```bash
python intrusion_detection/rf_train.py
python intrusion_detection/lightgbm_train.py
python intrusion_detection/cnn_train.py
```

### Launch Frontend

```bash
streamlit run frontend.py
```

---

## Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

All models are evaluated and compared using a common test dataset for fairness.

---

## Conclusion

This project demonstrates how **hybrid ML models** can enable **efficient, accurate intrusion detection** on **low-resource devices**. 

---


