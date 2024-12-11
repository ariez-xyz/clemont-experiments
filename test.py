import pandas as pd
import numpy as np

from aimon.discretization import Discretization
from aimon.monitor import Monitor

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

    discretization = Discretization(
            df,
            NBINS,
            'pred',
        )

    monitor = Monitor(discretization, "all")

    for index, row in df.iterrows():
        fair, cexs = monitor.observe(row, row_id=index)
        #print(fair, monitor.to_indices(cexs))
    print("monitor finished")


    true_positives = naive(df, 1/NBINS)
    print("computed naive solution")

    monitor_positives = sorted([p for p in monitor.unfair_pairs()])
    for true_positive in true_positives:
        if true_positive not in monitor_positives:
            print(f'ERROR: FALSE NEGATIVE. {true_positive} not in {monitor_positives}!!!!!!!!!!')
            break
    else:
        print("monitor delivered complete solution")
    print(f'Naive solution: {len(true_positives)}, monitor positives: {len(monitor_positives)}')

