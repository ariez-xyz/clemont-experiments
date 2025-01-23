import pandas as pd
import json
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    positives = data['positives']
    col = data['args']['pred']
    csvpath = data['args']['csvpath'][0]

df = pd.read_csv('../data/lcifr/predictions/train_features.csv')
pred_df = pd.read_csv(csvpath)

for pair in positives:
    idx1, idx2 = pair
    print(f"\nPair {idx1} - {idx2}:")
    
    # Create a DataFrame with both rows side by side, including prediction values
    comparison = pd.DataFrame({
        'column': ['Model prediction (income >50k?)'] + list(df.iloc[idx1].index),
        f'Row {idx1}': [pred_df.iloc[idx1][col]] + list(df.iloc[idx1].values),
        f'Row {idx2}': [pred_df.iloc[idx2][col]] + list(df.iloc[idx2].values)
    })
    
    # Print the formatted comparison
    print(comparison.to_string(index=False))
    print("-" * 80)
