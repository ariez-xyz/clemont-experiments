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
    df = pd.DataFrame(X_test.toarray())
    feature_cols = [col for col in df.columns]
    dtest = xgb.DMatrix(df[feature_cols])

    predictions = model.predict(dtest)
    
    if args.binary:
        output_df = pd.DataFrame({'pred': np.round(predictions).astype(int)})
    else:
        warnings.warn("Predictions will not be rounded to binary values.")
        output_df = pd.DataFrame({'pred': predictions})
    
    # Add original columns
    for col in feature_cols:
        output_df[col] = df[col]
    
    # Save predictions
    output_df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
