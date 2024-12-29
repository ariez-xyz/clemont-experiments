import numpy as np
import time

class Runner:
    def __init__(self, backend):
        self.backend = backend
        self.n_positives = 0
        self.n_true_positives = 0
        self.timings = []
        self.total_time = 0

    def get_backend_name(self):
        return self.backend.__class__.__name__

    def run(self, df, max_n=-1):
        all_cexs = []
        printed_progress = False

        for index, row in df.iterrows():
            if index == max_n:
                break
            start_iter_time = time.time()

            iter_cexs = self.backend.observe(row, row_id=index)

            self.n_positives += len(iter_cexs)

            # For unsound backends (BDD) filter false positives.
            if not self.backend.meta['is_sound']:
                if self.backend.meta['metric']!= "Linf":
                    print("warn: false positive filtering is currently only implemented for Linf distance")
                pred = self.backend.meta['decision_col']
                eps = self.backend.meta['epsilon']
                iter_cexs = self.filter_false_positives_Linf(iter_cexs, row, df, pred, eps)

            # Count remaining true positives after exact post-verification
            self.n_true_positives += len(iter_cexs)
            all_cexs.extend([(cex, index) for cex in iter_cexs])

            # Timing code.
            iter_time = time.time() - start_iter_time
            self.timings.append(iter_time)
            self.total_time += iter_time
            if self.total_time % 1 < 0.1:
                if not printed_progress:
                    printed_progress = True
                    print(f"{self.total_time:.2f}s: {index} items", end='\r')
            else:
                printed_progress = False

        print(f"Total time: {self.total_time:.2f} seconds")
        return all_cexs

    def filter_false_positives_Linf(self, cexs, row, df, pred, eps):
        nofp = [cex for cex in cexs if np.all(np.abs(df.loc[cex].drop(pred) - row.drop(pred)) < eps)]
        return nofp
