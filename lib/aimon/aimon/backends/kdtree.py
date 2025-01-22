from sklearn.neighbors import KDTree

from aimon.backends.base import BaseBackend
from aimon.backends.faiss import BruteForce

class KdTree(BaseBackend):
    def __init__(self, df, decision_col, epsilon, metric='infinity', batchsize=1000):
        if metric not in KDTree([[0]]).valid_metrics:
            raise NotImplementedError(f"invalid metric {metric}. valid metrics: {KDTree([[0]]).valid_metrics}")
        self.classes = df[decision_col].unique()
        self.df = df
        self.metric = metric
        self.batchsize = batchsize

        self.current_batch = 0
        self.bf = BruteForce(df, decision_col, epsilon)
        self.history = []
        self.histories = {c: [] for c in self.classes}

        self._meta = {
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": metric,
            "is_exact": True,
            "is_sound": True,
            "is_complete": True,
        }

    def observe(self, row, row_id=None):
        if len(self.history) <= self.batchsize:
            self.history.append(row)
            self.current_batch += 1
            return self.bf.observe(row, row_id)

        decision = row[self.meta["decision_col"]]

        if self.current_batch > self.batchsize: # Rebuild
            print(f"rebuilding at {len(self.history)}...")
            self.kdt = KDTree(self.history, metric=self.metric)
            self.bf = BruteForce(self.df, self.meta["decision_col"], self.meta["epsilon"])
            self.current_batch = 0

        cexs = self.bf.observe(row, row_id)

        # For each possible decision class, flip the current row's decision to that class
        # in order to find epsilon-close points with that (different) decision.
        for c in self.classes:
            # for the index matching the point's decision: skip search
            if c == decision:
                continue 
            row[self.meta["decision_col"]] = c
            kdt_res = self.kdt.query_radius([row], self.meta["epsilon"])
            cexs.extend(list(kdt_res[0]))

        row[self.meta["decision_col"]] = decision

        self.history.append(row)
        self.current_batch += 1

        return cexs

