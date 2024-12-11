import sys
import json
from collections import defaultdict
from dd.cudd import BDD

from .fairness import Linf_bdd

class Monitor:
    def __init__(self, discretization, collect_cex="none", print_stats=False):
        self.collect_cex = collect_cex
        self.discretization = discretization
        self.id_map = defaultdict(list)
        self.unfair_list = []

        self.bdd = BDD()
        self.bdd.declare(*self.discretization.bdd_vars)
    
        if print_stats:
            print(f"monitor running {len(self.discretization.bdd_vars)} variables", file=sys.stderr)

        self.history_bdd = self.bdd.false
        
        self.fairness_bdd = Linf_bdd(
                self.bdd, 
                discretization.cols, 
                discretization.decision, 
                self.discretization.vars_map,
                distances={cat: 0 for cat in discretization.categorical_cols} # Categories must be exact match, others count as similar if they are in neighboring bins
            )

        self.bdd.configure(reordering=False) # must be done after fairness bdd or it'll be very slow

    def hash_dict(self, d): # There may be better options but this is not as bad as it seems
        return json.dumps({k:v for k,v in d.items() if k.startswith('x_')}, sort_keys=True)

    def observe(self, row, row_id=None):
        if self.collect_cex == "all" and row_id == None:
            self.collect_cex = "none"
            print("warn: need a row_id to collect multiple counterexamples. disabling counterexample collection", file=sys.stderr)

        binned_row = self.discretization.bin_row(row)
        x_val, y_val = self.make_valuations(binned_row)
        
        if self.collect_cex == "all":
            self.id_map[self.hash_dict(x_val)].append(row_id)
        
        self.history_bdd = self.history_bdd | self.bdd.cube(x_val) # add current sample to history

        E = self.fairness_bdd & self.bdd.cube(y_val)
        E = E & self.history_bdd
        is_fair = (E == self.bdd.false)

        cex_vals = []
        if not is_fair:
            if self.collect_cex == "one":
                cex_vals = [self.bdd.pick(E)]
            elif self.collect_cex == "all":
                cex_vals += [cex for cex in self.bdd.pick_iter(E, care_vars=self.discretization.bdd_vars)]
                # Counterexample is provided by Cudd in the form of a valuation.
                # Remember pairs of row id and unfair valuation
                for cex in cex_vals: 
                    self.unfair_list.append((row_id, self.hash_dict(cex)))

        return is_fair, self.to_indices(cex_vals)

    def to_indices(self, valuations):
        ids = set()
        for v in valuations:
            ids |= set(self.id_map[self.hash_dict(v)])
        return ids

    def unfair_pairs(self):
        for yindex, valuation in self.unfair_list:
            for xindex in self.id_map[valuation]:
                if xindex >= yindex: break
                yield xindex, yindex
                
    def make_valuations(self, row):
        ret = []
        for p in 'xy':
            d = {}
            t = self.discretization.vars_map[p]
            for col in t.keys():
                for i, var in enumerate(t[col]):
                    if row[col] >> i & 1:
                        d[var] = True
                    else:
                        d[var] = False
            ret += [d]
        return ret

