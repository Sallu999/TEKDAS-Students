"""Train and evaluate the Student Dropout model.

Run after data_prep.py:
    python ml_model.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Deteksi ketersediaan model tree/ensemble (fallback graceful bila diblokir security policy)
try:
    from sklearn.ensemble import RandomForestClassifier
    MODEL_TYPE = "RandomForestClassifier"
except ImportError:
    from sklearn.linear_model import LogisticRegression
    MODEL_TYPE = "LogisticRegression"

from data_prep import (
    BASE_DIR,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET,
)


def build_pipeline() -> Pipeline:
    """Membangun Scikit-learn Pipeline preprocessing dan klasifikasi."""
    if MODEL_TYPE == "RandomForestClassifier":
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        classifier = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        # Standard scaler diperlukan untuk Logistic Regression
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", classifier),
    ])


def main():
    train_path = BASE_DIR / "student_train.csv"
    test_path = BASE_DIR / "student_test.csv"

    if not train_path.exists() or not test_path.exists():
        print("Data train/test belum ditemukan. Menjalankan data_prep.py...")
        from data_prep import main as run_data_prep
        run_data_prep()

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET]

    pipeline = build_pipeline()
    print(f"Melatih model ({MODEL_TYPE})...")
    pipeline.fit(X_train, y_train)

    classes = list(pipeline.named_steps["model"].classes_)
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    metrics = {
        "model_type": MODEL_TYPE,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
    }

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, labels=classes, digits=3, zero_division=0))
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Accuracy  : {acc:.3f}")
    print(f"Macro F1  : {macro_f1:.3f}")
    print(f"Weight F1 : {weighted_f1:.3f}")

    # Simpan pipeline
    joblib.dump(pipeline, BASE_DIR / "student_dropout_pipeline.pkl")

    # Hitung feature importance
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        raw_importances = pipeline.named_steps["model"].feature_importances_
    else:
        # Untuk linear model, gunakan rerata magnitudo koefisien absolut
        coefs = np.abs(pipeline.named_steps["model"].coef_).mean(axis=0)
        raw_importances = coefs / coefs.sum()

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": raw_importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(BASE_DIR / "feature_importance.csv", index=False)

    # Simpan prediksi test set dengan ID mahasiswa
    pred_df = test[[ID_COLUMN, TARGET]].copy()
    pred_df["predicted_target"] = y_pred
    for i, cls_name in enumerate(classes):
        pred_df[f"prob_{cls_name.lower()}"] = y_probs[:, i]
    pred_df.to_csv(BASE_DIR / "student_predictions.csv", index=False)

    # Simpan metrik
    with open(BASE_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts:")
    print("  - student_dropout_pipeline.pkl")
    print("  - feature_importance.csv")
    print("  - student_predictions.csv")
    print("  - model_metrics.json")


if __name__ == "__main__":
    main()
