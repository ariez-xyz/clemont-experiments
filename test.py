import pandas as pd
import numpy as np

from aimon.discretization import Discretization
from aimon.monitor import Monitor

np.set_printoptions(suppress=True)

EXPECTED =  [(1, 127), (4, 111), (4, 122), (6, 137), (7, 90), (8, 83), (8, 96), (8, 126), (8, 152), (9, 32), (13, 15), (13, 28), (13, 113), (14, 62), (14, 87), (15, 107), (16, 106), (16, 118), (17, 138), (17, 156), (18, 80), (19, 27), (19, 161), (21, 152), (21, 158), (22, 49), (22, 83), (22, 96), (23, 165), (23, 173), (24, 75), (25, 80), (25, 147), (26, 36), (26, 44), (26, 178), (27, 177), (28, 96), (28, 107), (28, 153), (28, 158), (30, 117), (30, 161), (32, 36), (32, 44), (32, 100), (34, 78), (34, 82), (34, 120), (34, 135), (35, 152), (37, 170), (38, 48), (38, 124), (39, 79), (41, 55), (41, 124), (42, 103), (43, 181), (47, 48), (48, 67), (48, 197), (49, 117), (54, 77), (56, 60), (56, 64), (62, 111), (62, 136), (62, 148), (63, 65), (63, 126), (63, 179), (63, 198), (65, 109), (65, 155), (65, 170), (65, 188), (69, 70), (69, 128), (69, 137), (69, 148), (70, 160), (71, 76), (71, 130), (71, 178), (73, 193), (74, 122), (75, 161), (76, 150), (79, 114), (79, 135), (79, 161), (81, 178), (82, 139), (83, 113), (83, 150), (87, 148), (90, 116), (91, 132), (92, 134), (96, 99), (96, 113), (96, 170), (97, 126), (97, 198), (98, 159), (100, 193), (105, 177), (106, 173), (107, 125), (109, 126), (112, 167), (116, 165), (120, 139), (120, 149), (122, 139), (127, 153), (128, 173), (128, 196), (129, 163), (131, 193), (134, 183), (134, 192), (135, 165), (135, 168), (135, 178), (138, 192), (140, 150), (144, 172), (148, 196), (149, 150), (149, 175), (150, 172), (154, 158), (156, 190), (156, 192), (161, 165), (161, 191), (163, 184), (165, 180), (173, 180), (187, 195)]

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
    df['pred'] = (df['pred'] > 0.5).astype(int)

    discretization = Discretization(
            df,
            NBINS,
            'pred',
        )

    monitor = Monitor(discretization, "all")

    all_cexs = []
    for index, row in df.iterrows():
        fair, iter_cexs = monitor.observe(row, row_id=index)
        all_cexs.extend([(cex, index) for cex in iter_cexs])

    true_positives = naive(df, 1/NBINS)

    monitor_positives = sorted(all_cexs)

    assert monitor_positives == EXPECTED, f"deviated from expected solution\ngot:{monitor_positives}\nexpected:{EXPECTED}"
    print("monitor solution is as expected")

    for true_positive in true_positives:
        if true_positive not in monitor_positives:
            print(f'ERROR: FALSE NEGATIVE. {true_positive} not in {monitor_positives}!!!!!!!!!!')
            break
    else:
        print("monitor delivered complete solution")
    print(f'Naive solution: {len(true_positives)}, monitor positives: {len(monitor_positives)}')

