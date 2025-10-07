#!/usr/bin/env python3
# train_predict_save.py
import sys
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

TARGET_CANDIDATES = ["income", "class"]  # common names in Adult datasets

def detect_target_column(df: pd.DataFrame) -> str:
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find target column. Looked for {TARGET_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )

def load_data(path: str):
    df = pd.read_csv(path)

    # Strip whitespace in all object columns (Adult often has leading spaces)
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    target_col = detect_target_column(df)
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])

    # Identify column types (keep all rows; treat '?' as just another category)
    numeric_cols: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols: List[str] = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preprocessing as the model will see: scale numerics, one-hot categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,  # helps feature names show their transformer
    )

    return df, X, y, preprocessor

def train_with_split(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=0,
                n_jobs=-1
            )),
        ]
    )

    clf.fit(X_train, y_train)

    # Evaluate on held-out test
    y_proba_test = clf.predict_proba(X_test)
    y_pred_test = clf.predict(X_test)

    print("=== Held-out evaluation (train/test split) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    try:
        print(f"Log loss: {log_loss(y_test, y_proba_test):.4f}")
    except ValueError:
        # If a class is entirely missing in y_test (rare with stratify), skip log loss
        pass
    print("Classification report:\n", classification_report(y_test, y_pred_test))

    return clf

def save_full_predictions(clf: Pipeline, df_original: pd.DataFrame, X_full: pd.DataFrame, y_full: pd.Series,
                          probs_path="predictions_with_probs.csv"):
    # Predict on FULL dataset in ORIGINAL order
    proba_full = clf.predict_proba(X_full)
    pred_full = clf.predict(X_full)

    classes = list(clf.named_steps["clf"].classes_)  # original string labels (e.g., '<=50K', '>50K')

    out = pd.DataFrame({
        "row_id": df_original.index.to_series().values,  # sanity-check alignment
        "predicted_class": pred_full,
        "true_class": y_full,
    })

    # Add per-class probability columns using original class labels in the column names
    for j, cls in enumerate(classes):
        out[f"prob_{cls}"] = proba_full[:, j]

    out.to_csv(probs_path, index=False)
    print(f"Saved probabilities aligned with input order -> {probs_path}")

def save_full_numeric_inputs(clf: Pipeline, X_full: pd.DataFrame, df_original: pd.DataFrame,
                             inputs_path="inputs_numeric.csv"):
    # Transform FULL dataset with the fitted preprocessor (fitted on train only)
    preproc = clf.named_steps["preproc"]  # ColumnTransformer
    X_num = preproc.transform(X_full)

    # Get feature names exactly as the model sees them (scaled numeric + one-hot)
    try:
        feature_names = preproc.get_feature_names_out()
    except AttributeError:
        # Fallback: generic names if running older sklearn
        feature_names = [f"f_{i}" for i in range(X_num.shape[1])]

    numeric_df = pd.DataFrame(X_num, columns=feature_names)
    numeric_df.insert(0, "row_id", df_original.index.to_series().values)  # preserve alignment

    numeric_df.to_csv(inputs_path, index=False)
    print(f"Saved numerical design matrix (model inputs) -> {inputs_path}")
    print(f"Shape: {numeric_df.shape[0]} rows x {numeric_df.shape[1]} columns")

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_predict_save.py <path_to_adult_csv> [probs_csv] [inputs_csv]")
        sys.exit(1)

    path = sys.argv[1]
    probs_csv = sys.argv[2] if len(sys.argv) >= 3 else "predictions_with_probs.csv"
    inputs_csv = sys.argv[3] if len(sys.argv) >= 4 else "inputs_numeric.csv"

    df, X, y, preprocessor = load_data(path)
    clf = train_with_split(X, y, preprocessor)

    # Final pass over the entire dataset (original order)
    save_full_predictions(clf, df, X, y, probs_path=probs_csv)
    save_full_numeric_inputs(clf, X, df, inputs_path=inputs_csv)

if __name__ == "__main__":
    main()
