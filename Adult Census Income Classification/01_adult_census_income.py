"""
Adult Census Income Classification
Compares 5 classification models on the UCI Adult Census Income dataset.
Dataset is fetched automatically via OpenML - no manual download needed.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -------------------- 1. Load data --------------------
print("Fetching Adult Census Income dataset from OpenML...")
data = fetch_openml(name="adult", version=2, as_frame=True)
df = data.frame.copy()

# Target column is 'class' in this OpenML version
target_col = "class"
df = df.dropna()  # simplest handling of missing values for a quick pipeline
print(f"Dataset shape after dropping NA: {df.shape}")

# -------------------- 2. Encode categoricals --------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# Label-encode target (>50K / <=50K)
y = LabelEncoder().fit_transform(y)

# One-hot encode categorical features, keep numeric as-is
X = pd.get_dummies(X, drop_first=True)

# -------------------- 3. Train/test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 4. Models --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=False),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    # Tree-based models don't need scaling; linear/SVM do
    if name in ("Logistic Regression", "SVM (RBF)"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
    print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# -------------------- 5. Summary --------------------
results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print("\n===== Model Comparison (sorted by F1) =====")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest performing model: {best_model_name}")