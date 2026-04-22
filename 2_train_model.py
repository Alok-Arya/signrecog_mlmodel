import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH = "dataset/landmarks.csv"
MODEL_PATH = "models/sign_model.pkl"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
data = pd.read_csv(DATA_PATH, header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print("\n📦 Dataset shape:", X.shape)
print("📌 Number of classes:", len(np.unique(y)))

# ─────────────────────────────────────────────
# DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n📊 Missing values check:\n", data.isnull().sum().sum())

print("\n📊 Data statistics:\n", data.describe())

# ─────────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────────
class_counts = pd.Series(y).value_counts()

print("\n📊 Class Distribution:\n", class_counts)

plt.figure(figsize=(10,5))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("reports/class_distribution.png")
plt.show()

# ─────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📌 Train size:", X_train.shape)
print("📌 Test size:", X_test.shape)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

# ─────────────────────────────────────────────
# TRAIN TIME
# ─────────────────────────────────────────────
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

print("\n⏱ Training time:", end_train - start_train, "seconds")

# ─────────────────────────────────────────────
# PREDICTION TIME
# ─────────────────────────────────────────────
start_pred = time.time()
y_pred = model.predict(X_test)
end_pred = time.time()

print("⏱ Prediction time:", end_pred - start_pred, "seconds")

# ─────────────────────────────────────────────
# ACCURACY
# ─────────────────────────────────────────────
train_acc = model.score(X_train, y_train)
test_acc = accuracy_score(y_test, y_pred)

print("\n🎯 Train Accuracy:", train_acc)
print("🎯 Test Accuracy:", test_acc)

# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────
cv_scores = cross_val_score(model, X, y, cv=5)

print("\n📊 Cross Validation Scores:", cv_scores)
print("📊 Mean CV Accuracy:", cv_scores.mean())

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("reports/confusion_matrix.png")
plt.show()

# ─────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_df.to_csv("reports/classification_report.csv")

print("\n📄 Classification Report saved!")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = model.feature_importances_
top_idx = np.argsort(importances)[-20:]

plt.figure(figsize=(10,6))
plt.barh(range(len(top_idx)), importances[top_idx])
plt.title("Top 20 Feature Importance")
plt.savefig("reports/feature_importance.png")
plt.show()

# ─────────────────────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────────────────────
errors = (y_test != y_pred).sum()
print("\n❌ Total Wrong Predictions:", errors)

# ─────────────────────────────────────────────
# MODEL SIZE
# ─────────────────────────────────────────────
model_size = os.path.getsize(MODEL_PATH) / 1024 if os.path.exists(MODEL_PATH) else 0

print("💾 Model size (KB):", model_size)

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)

print("\n💾 Model saved at:", MODEL_PATH)
print("📁 All reports saved in /reports folder")