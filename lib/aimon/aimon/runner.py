import numpy as np

class Runner:
    def __init__(self, backend):
        self.backend = backend
        self.n_positives = 0
        self.n_true_positives = 0

    def run(self, df):
        all_cexs = []
        for index, row in df.iterrows():
            iter_cexs = self.backend.observe(row, row_id=index)

            # For unsound backends (BDD) filter false positives.
            if self.backend.meta['is_sound'] == False:
                if self.backend.meta['metric'] != "Linf":
                    print("warn: false positive filtering is currently only implemented for Linf distance")
                pred = self.backend.meta['decision_col']
                eps = self.backend.meta['epsilon']
                iter_cexs = self.filter_false_positives_Linf(iter_cexs, row, df, pred,eps)

            all_cexs.extend([(cex, index) for cex in iter_cexs])
        return all_cexs

    def filter_false_positives_Linf(self, cexs, row, df, pred, eps):
        nofp = [cex for cex in cexs if np.all(np.abs(df.loc[cex].drop(pred) - row.drop(pred)) < eps)]
        self.n_positives += len(cexs)
        self.n_true_positives += len(nofp)
        return nofp

