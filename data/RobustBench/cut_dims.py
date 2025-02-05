import pandas as pd

i1="predictions/imagenet-Standard_R50.csv"
i2="predictions/imagenet-Standard_R50-adv.csv"

pred="pred"
samplecols=map(int, "8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 65536 131072".split())

# Shamelessly ripped from run_on_csv.py
dfs = []
for arg in [i1,i2]:
    # Handle potential newline-separated paths
    paths = arg.split('\n')
    for path in paths:
        path = path.strip()
        if not path:
            continue
        df = pd.read_csv(path)
        # Ensure consistent column structure across dataframes
        if dfs and df.shape[1] != dfs[0].shape[1]:
            raise ValueError(f"CSV at {path} has incompatible column structure")
        dfs.append(df)
df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"loaded data of shape {df.shape}.")

for sample_cols in samplecols:
    non_pred_cols = df.columns.drop(pred)
    sampled_columns = non_pred_cols.to_series().sample(n=sample_cols, random_state=0)
    keep = pd.concat([sampled_columns, pd.Series([pred])])
    sampled_df = df[keep].copy()
    print(f"keeping columns: {list(keep)}")
    print(f"new shape is {sampled_df.shape}")
    sampled_df.to_csv(f"predictions/imagenet-Standard_R50-combined-{sample_cols}d.csv", index=False)
