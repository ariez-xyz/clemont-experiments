import faiss
import numpy as np

from aimon.backends.base import BaseBackend
from typing import List

class BruteForce(BaseBackend):
    def __init__(self, df, decision_col, epsilon):
        self.dim = df.shape[1] - 1 # kNN algo is blind to the decision column

        # Create separate indices for each unique class
        self.indices = {}
        for class_val in df[decision_col].unique():
            flat_index = faiss.IndexFlat(self.dim, faiss.METRIC_Linf)
            with_custom_ids = faiss.IndexIDMap(flat_index) # This decorator adds support for add_with_ids()
            self.indices[class_val] = with_custom_ids
        print(f"initialized {len(self.indices)} indices. eps={epsilon}")

        self.epsilon = epsilon
        self.decision_col = decision_col
        self.radius_query_ks = []
        self._meta = {
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": "Linf",
            "is_exact": True,
            "is_sound": True,
            "is_complete": False,
        }

    def observe(self, row, row_id=None) -> List[int]:
        cexs = []
        row_data = np.array(row.drop(self.decision_col)).reshape(1, -1)

        for decision, idx in self.indices.items():
            # for the index matching the point's decision: skip search, instead add point
            if decision == row[self.decision_col]:
                idx.add_with_ids(row_data, [row_id]) # pass explicit id - the automatically assigned sequential ids are only unique within each index
                continue 

            query_fn = lambda k: idx.search(row_data, k)
            _, indices = self.emulate_range_query(query_fn, self.epsilon)
            cexs.extend(indices)

        return cexs
