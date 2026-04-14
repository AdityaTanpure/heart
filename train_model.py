import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

print("Loading feature dataset...")
df = pd.read_csv("features.csv")

X = df.drop("label", axis=1)
y = df["label"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Building Random Forest model...")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("Evaluating model...")

# Get probabilities
probs = pipeline.predict_proba(X_test)[:, 1]

# Adjust threshold
threshold = 0.3
preds = (probs >= threshold).astype(int)

print("\n==============================")
print(f"Random Forest Performance (Threshold = {threshold})")
print("==============================")

print("\nAccuracy:", accuracy_score(y_test, preds))
print("ROC-AUC Score:", roc_auc_score(y_test, probs))

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, preds))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Murmur Detection")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()


# Save model
joblib.dump(pipeline, "murmur_model.joblib")

print("\nModel saved as murmur_model.joblib")
