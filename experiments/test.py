import pandas as pd
import numpy as np

from aimon.backends.bdd import BDD
from aimon.backends.faiss import BruteForce
from aimon.runner import Runner

np.set_printoptions(suppress=True)

def naive(df, epsilon, num_columns):
    unfair_pairs = []
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            row_i, row_j = df.iloc[i], df.iloc[j]
            
            if abs(row_i['pred'] - row_j['pred']) > epsilon:
                if all(abs(row_i[f'Column_{k}'] - row_j[f'Column_{k}']) <= epsilon for k in range(1, num_columns)):
                    unfair_pairs.append((i, j))
    
    return unfair_pairs

def pprint_pair(df, i, j, eps):
    print(f'\nrow {i}\trow {j}\tdiff\t<{eps}?')
    for col in range(len(df.columns)):
        val_i = df.iloc[i, col] # Format these to strings with 2 decimals
        val_j = df.iloc[j, col]
        diff = abs(val_i - val_j)
        is_close = diff < eps
        print(f"{val_i:.2f}\t{val_j:.2f}\t{diff:.2f}\t{is_close}")


if __name__ == "__main__":
    num_rows = 200
    num_columns = 10
    NBINS = 4
    column_names = ['pred'] + [f'Column_{i}' for i in range(1, num_columns)]
    np.random.seed(42)

    data = np.random.uniform(0, 1, size=(num_rows, num_columns))
    df = pd.DataFrame(data, columns=column_names)
    df['pred'] = (df['pred'] > 0.5).astype(int) # Make decision binary coinflip

    bdd = BDD(
        data_sample=df,
        n_bins=NBINS,
        decision_col='pred',
        collect_cex=True
    )

    bf = BruteForce(df, 'pred', 1/NBINS)

    #runner = Runner(bdd)
    runner = Runner(bf)

    monitor_positives = sorted(runner.run(df))
    naive_positives   = sorted(naive(df, 1/NBINS, num_columns))

    if monitor_positives != naive_positives:
        print("monitor returned different result from naive solution")
        print("naive solution:")
        print(naive_positives)
        print("monitor solution:")
        print(monitor_positives)
        print("-------------\nmonitor pairs:")
        for pair in monitor_positives:
            pprint_pair(df, pair[0], pair[1], 1/NBINS)
        print("-------------\nnaive pairs:")
        for pair in naive_positives:
            pprint_pair(df, pair[0], pair[1], 1/NBINS)
        exit(1)

    print(f'correct solution {monitor_positives} obtained from both implementations')
    print(f'monitor gave {runner.n_true_positives} true positives out of {runner.n_positives}')

