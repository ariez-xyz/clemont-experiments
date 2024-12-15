import sys
import json
from collections import defaultdict
from dd.cudd import BDD

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
        
        self.fairness_bdd = self.Linf_bdd(
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

    # Create BDD over 2n variables. BDD should be True if the two inputs represent a biased decision
    # That can mean anything you want it to. A practical limitation is being able to build a BDD from the map (enumerating all input pairs is quickly infeasible)
    def Linf_bdd(self, bdd, cols, decision, vars_map, distances={}):
        D = bdd.true
        warned = False
        
        def same(col, j, nbits):
            return f'(( x_{col}_{nbits-j-1} &  y_{col}_{nbits-j-1}) | (!x_{col}_{nbits-j-1} & !y_{col}_{nbits-j-1}))'
        def diff(col, j, nbits):
            return f'(( x_{col}_{nbits-j-1} & !y_{col}_{nbits-j-1}) | (!x_{col}_{nbits-j-1} &  y_{col}_{nbits-j-1}))'
        def one(var, col, j, nbits):
            return f' {var}_{col}_{nbits-j-1}'
        def zero(var, col, j, nbits):
            return f'!{var}_{col}_{nbits-j-1}'
        
        
        for col in cols:
            col_flas = []
            col_nbits = len(vars_map['x'][col])
        
            # same-case
            col_flas.append(f"({' & '.join([same(col, j, col_nbits) for j in range(col_nbits)])})")

            if col == decision: # decision bit must always be different
                D = D & ~bdd.add_expr(col_flas[0])
                continue
            
            # default: neighboring bins count as similar
            if col not in distances.keys() or distances[col] == 1:
                for order in [('x', 'y'), ('y', 'x')]:
                    for k in range(col_nbits):
                        terms = []
                
                        # first k-1 bits match
                        for j in range(k):
                            terms.append(same(col, j, col_nbits))
                
                        # first has suffix 100000... 
                        terms.append(one(order[0], col, k, col_nbits))
                        for j in range(k+1, col_nbits):
                            terms.append(zero(order[0], col, j, col_nbits))
                
                        # second has suffix 011111...
                        terms.append(zero(order[1], col, k, col_nbits))
                        for j in range(k+1, col_nbits):
                            terms.append(one(order[1], col, j, col_nbits))
                        
                        #print("\n","\n      &\t".join(terms), sep='\t')
                        col_flas.append(f'({" & ".join(terms)})')

            if col in distances.keys() and distances[col] not in {0,1}:
                if not warned:
                    warned = True
                    print(f'warning: currently similarity BDD only supports distances 0,1! defaulted {col} to 0.', file=sys.stderr)
                
            # at this point col_flas contains a list of ways in that  | x_col - y_col | <= distances[col]
            # OR these into a single fla
            col_matches = f"({'   |   '.join(col_flas)})"

            
            # the similarity BDD wants every col to be close
            D = D & bdd.add_expr(col_matches)

        return D

