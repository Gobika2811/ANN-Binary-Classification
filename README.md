# 🧠 ANN Binary Classification

A complete **Artificial Neural Network (ANN)** binary classification project built using **scikit-learn's MLPClassifier** on the `make_moons` dataset. This project demonstrates end-to-end ML workflow — from data generation to model evaluation with rich visualizations.

---

## 📌 Project Overview

| Property | Details |
|---|---|
| **Algorithm** | Artificial Neural Network (MLP) |
| **Task** | Binary Classification |
| **Dataset** | `make_moons` (sklearn synthetic dataset) |
| **Total Samples** | 500 |
| **Train / Test Split** | 400 / 100 (80/20) |
| **Architecture** | Input → 16 → 8 → Output |
| **Activation** | ReLU |
| **Optimizer** | Adam |

---

## 📁 Project Structure

```
ANN_Binary_Classification/
│
├── main.py                  # Main training & evaluation script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
└── outputs/
    ├── dataset.png          # make_moons dataset visualization
    ├── decision_boundary.png # ANN decision boundary plot
    ├── confusion_matrix.png  # Confusion matrix heatmap
    ├── loss_curve.png        # Training loss curve
    └── results.txt          # Saved evaluation metrics
```

---

## 🏗️ Model Architecture

```
Input Layer  →  2 neurons  (Feature 1, Feature 2)
Hidden Layer 1 → 16 neurons (ReLU activation)
Hidden Layer 2 →  8 neurons (ReLU activation)
Output Layer  →  1 neuron  (Binary: Class 0 / Class 1)
```

**Training Config:**
- Solver: `Adam`
- Learning Rate: `0.01`
- Max Iterations: `2000`
- Early Stopping: ✅ Enabled (patience = 30)
- Validation Fraction: `10%`

---

## 📊 Model Evaluation Results

| Metric | Score |
|---|---|
| ✅ **Accuracy** | **94.00%** |
| 🎯 Precision | 0.9231 |
| 🔁 Recall | 0.9600 |
| ⚖️ F1-Score | 0.9412 |
| 🔒 Specificity | 0.9200 |
| 📐 MCC | 0.8807 |
| 📉 RMSE | 0.2449 |

### Confusion Matrix

```
              Predicted 0   Predicted 1
Actual 0   [     46              4     ]
Actual 1   [      2             48     ]
```

- **True Negatives (TN):** 46
- **False Positives (FP):** 4
- **False Negatives (FN):** 2
- **True Positives (TP):** 48

---

## 📈 Visualizations

The following plots are auto-generated in the `outputs/` folder:

1. **Dataset Plot** — make_moons distribution (Class 0 vs Class 1)
2. **Decision Boundary** — ANN's learned classification boundary on test data
3. **Confusion Matrix** — Heatmap of prediction results
4. **Loss Curve** — Training loss over iterations

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Gobika2811/ANN-Binary-Classification.git
cd ANN-Binary-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python main.py
```

> All results and graphs will be saved automatically in the `outputs/` folder.

---

## 🔧 Requirements

```
numpy
matplotlib
scikit-learn
```

> Python **3.8+** recommended

---

## 🧪 Key Concepts Covered

- ✅ Synthetic dataset generation (`make_moons`)
- ✅ Train-test split with stratification
- ✅ Feature scaling (`StandardScaler`)
- ✅ ANN model building with `MLPClassifier`
- ✅ Early stopping to prevent overfitting
- ✅ 8 evaluation metrics (Accuracy, Precision, Recall, F1, Specificity, MCC, RMSE, Confusion Matrix)
- ✅ Decision boundary visualization
- ✅ Loss curve plotting

---
