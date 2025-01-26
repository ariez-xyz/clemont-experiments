import time

from sklearn.neighbors import KDTree

from aimon.backends.base import BaseBackend
from aimon.backends.faiss import BruteForce

class KdTree(BaseBackend):
    def __init__(self, df, decision_col, epsilon, metric='infinity', batchsize=1000):
        if metric not in KDTree([[0]]).valid_metrics:
            raise NotImplementedError(f"invalid metric {metric}. valid metrics: {KDTree([[0]]).valid_metrics}")

        self.classes = df[decision_col].unique()
        self.df = df
        self.batchsize = batchsize

        self.current_batch = 0
        self.bf = BruteForce(df, decision_col, epsilon, metric)
        self.history = []
        self.histories = {c: [] for c in self.classes}

        self._meta = {
            "kdt_time": 0,
            "bf_time": 0,
            "index_time": 0,
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": metric,
            "batchsize": batchsize,
            "is_exact": True,
            "is_sound": True,
            "is_complete": True,
        }

    def observe(self, row, row_id=None):
        if len(self.history) < self.batchsize:
            self.history.append(row)
            self.current_batch += 1
            return self.bf.observe(row, row_id)

        decision = row[self.meta["decision_col"]]

        if self.current_batch >= self.batchsize: # Rebuild
            print(f"rebuilding at {len(self.history)}...")
            st = time.time()
            self.kdt = KDTree(self.history, metric=self.meta["metric"])
            self.bf = BruteForce(self.df, self.meta["decision_col"], self.meta["epsilon"], self.meta["metric"])
            self.meta["index_time"] += time.time() - st
            self.current_batch = 0

        # First identify close points within the current batch using brute force
        st = time.time()
        cexs = self.bf.observe(row, row_id)
        self.meta["bf_time"] += time.time() - st

        # Now query the previous batches which are stored in the kd-tree
        for c in self.classes:
            # For each possible decision class, flip the current row's decision to that class
            # in order to find epsilon-close points with that (different) decision.
            if c == decision:
                continue # skip search for points with same decision
            row[self.meta["decision_col"]] = c
            st = time.time()
            kdt_res = self.kdt.query_radius([row], self.meta["epsilon"])
            self.meta["kdt_time"] += time.time() - st
            cexs.extend(list(kdt_res[0]))

        row[self.meta["decision_col"]] = decision # Restore point to correct decision.

        self.history.append(row)
        self.current_batch += 1

        return cexs

