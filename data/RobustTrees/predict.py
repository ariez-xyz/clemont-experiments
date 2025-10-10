import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from sklearn.datasets import load_svmlight_file

def parse_args():
    parser = argparse.ArgumentParser(description='Run predictions using trained XGBoost model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to XGBoost model checkpoint (.model file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save predictions CSV')
    parser.add_argument('--binary', action='store_true', default=False,
                        help='Binary classification task? False for multi-class')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the model
    model = xgb.Booster()
    model.load_model(args.model_path)
    
    # Load test data
    X_test, y_test = load_svmlight_file(args.data)
    df = pd.DataFrame(X_test.toarray(), copy=False)  # avoid copying the array to reduce fragmentation
    feature_cols = list(df.columns)
    dtest = xgb.DMatrix(df[feature_cols])

    raw_predictions = model.predict(dtest)

    class_labels = np.unique(y_test)
    if args.binary and class_labels.size > 2:
        warnings.warn(
            "--binary flag set but more than two unique labels detected. Proceeding with multi-class handling."
        )
        args.binary = False

    if args.binary:
        # Ensure predictions are probabilities for the positive class
        if raw_predictions.ndim == 2:
            if raw_predictions.shape[1] == 2:
                prob_pos = raw_predictions[:, 1]
            else:
                raise ValueError(
                    f"Unexpected prediction shape {raw_predictions.shape} for binary task."
                )
        else:
            prob_pos = raw_predictions.reshape(-1)

        # Assume positive class is the larger of the sorted labels
        sorted_labels = np.sort(class_labels)
        if sorted_labels.size == 0:
            sorted_labels = np.array([0, 1])
        elif sorted_labels.size == 1:
            sorted_labels = np.append(sorted_labels, 1)

        neg_label, pos_label = sorted_labels[0], sorted_labels[-1]
        prob_neg = 1.0 - prob_pos
        # Clip to avoid numerical artefacts
        prob_neg = np.clip(prob_neg, 0.0, 1.0)
        prob_pos = np.clip(prob_pos, 0.0, 1.0)

        pred_labels = np.where(prob_pos >= 0.5, pos_label, neg_label)
        prob_df = pd.DataFrame(
            {
                f"prob_{neg_label}": prob_neg,
                f"prob_{pos_label}": prob_pos,
            }
        )
    else:
        # Multi-class: expect probability matrix
        if raw_predictions.ndim == 1:
            raise ValueError(
                "Expected probability matrix for multi-class prediction, received 1D array instead."
            )
        if class_labels.size == 0:
            class_labels = np.arange(raw_predictions.shape[1])
        else:
            class_labels = np.sort(class_labels)

        if raw_predictions.shape[1] != class_labels.size:
            # Fall back to positional labels if mismatch
            warnings.warn(
                "Mismatch between number of predicted classes and label set; using positional labels 0..k-1."
            )
            class_labels = np.arange(raw_predictions.shape[1])

        pred_indices = np.argmax(raw_predictions, axis=1)
        pred_labels = np.array([class_labels[i] for i in pred_indices])

        prob_columns = {
            f"prob_{class_labels[idx]}": raw_predictions[:, idx]
            for idx in range(len(class_labels))
        }
        prob_df = pd.DataFrame(prob_columns)

    output_df = pd.DataFrame({"pred": pred_labels, "label": y_test})
    output_df = pd.concat([output_df, prob_df, df[feature_cols]], axis=1)

    output_df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
