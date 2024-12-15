import pandas as pd
import numpy as np

from aimon.backends.bdd import BDD
from aimon.runner import Runner

np.set_printoptions(suppress=True)

def naive(df, epsilon):
    unfair_pairs = []
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            row_i, row_j = df.iloc[i], df.iloc[j]
            
            if abs(row_i['pred'] - row_j['pred']) > epsilon:
                if all(abs(row_i[f'Column_{k}'] - row_j[f'Column_{k}']) <= epsilon for k in range(1, num_columns)):
                    unfair_pairs.append((i, j))
    
    return unfair_pairs


if __name__ == "__main__":
    num_rows = 200
    num_columns = 10
    NBINS = 4
    column_names = ['pred'] + [f'Column_{i}' for i in range(1, num_columns)]
    np.random.seed(42)

    data = np.random.uniform(0, 1, size=(num_rows, num_columns))
    df = pd.DataFrame(data, columns=column_names)
    df['pred'] = (df['pred'] > 0.5).astype(int) # Make decision binary coinflip

    backend = BDD(
        data_sample=df,
        n_bins=NBINS,
        decision_col='pred',
        collect_cex=True
    )

    runner = Runner(backend)

    monitor_positives = sorted(runner.run(df))
    naive_positives   = sorted(naive(df, 1/NBINS))

    assert monitor_positives == naive_positives, f"monitor returned different result from naive solution\nmonitor:{monitor_positives}\nnaive:{naive_positives}"

    print(f'correct solution {monitor_positives} obtained from both implementations')
    print(f'monitor gave {runner.n_true_positives} true positives out of {runner.n_positives}')

