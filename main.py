
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    mean_squared_error
)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Generate dataset
X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)

print("=" * 45)
print(" ANN BINARY CLASSIFICATION USING make_moons")
print("=" * 45)
print("Total samples:", len(X))
print("Total features:", X.shape[1])
print("Class 0 samples:", np.sum(y == 0))
print("Class 1 samples:", np.sum(y == 1))

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Keep original coordinates for plotting
X_test_original = X_test.copy()

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build ANN model
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30
)

# 5. Train model
model.fit(X_train, y_train)
print("\nANN model trained successfully!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate 8 evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp) if (tn + fp) > 0 else 0
)
mcc = matthews_corrcoef(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 8. Print evaluation results
print("\n" + "=" * 45)
print("         MODEL EVALUATION RESULTS")
print("=" * 45)

print(f"1. Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"2. Precision    : {precision:.4f}")
print(f"3. Recall       : {recall:.4f}")
print(f"4. F1-Score     : {f1:.4f}")

print("\n5. Confusion Matrix:")
print(cm)
print(f"   True Negative  : {tn}")
print(f"   False Positive : {fp}")
print(f"   False Negative : {fn}")
print(f"   True Positive  : {tp}")

print(f"\n6. Specificity   : {specificity:.4f}")
print(f"7. MCC           : {mcc:.4f}")
print(f"8. RMSE          : {rmse:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Save text results
with open("outputs/results.txt", "w", encoding="utf-8") as file:
    file.write("ANN BINARY CLASSIFICATION RESULTS\n")
    file.write("=" * 40 + "\n")
    file.write(f"Dataset samples: {len(X)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n\n")

    file.write(f"1. Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    file.write(f"2. Precision: {precision:.4f}\n")
    file.write(f"3. Recall: {recall:.4f}\n")
    file.write(f"4. F1-Score: {f1:.4f}\n")
    file.write(f"5. Confusion Matrix:\n{cm}\n")
    file.write(f"6. Specificity: {specificity:.4f}\n")
    file.write(f"7. MCC: {mcc:.4f}\n")
    file.write(f"8. RMSE: {rmse:.4f}\n\n")
    file.write(classification_report(y_test, y_pred, zero_division=0))

# 10. Original dataset graph
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap="coolwarm",
    edgecolors="black", s=40
)
plt.title("make_moons Dataset Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("outputs/dataset.png", dpi=300)
plt.show()

# 11. ANN decision boundary graph
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = model.predict(grid_scaled).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(
    xx, yy, Z,
    alpha=0.3, cmap="coolwarm"
)
plt.scatter(
    X_test_original[:, 0],
    X_test_original[:, 1],
    c=y_test, cmap="coolwarm",
    edgecolors="black", s=45
)
plt.title("ANN Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("outputs/decision_boundary.png", dpi=300)
plt.show()

# 12. Confusion matrix graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1"]
)
display.plot(cmap="Blues", values_format="d")
plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# 13. Training loss curve
plt.figure(figsize=(8, 6))
plt.plot(model.loss_curve_)
plt.title("ANN Training Loss Curve")
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=300)
plt.show()

print("\n" + "=" * 45)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("All graphs and results saved in outputs folder.")
print("=" * 45)
