import sys

# Create BDD over 2n variables. BDD should be True if the two inputs represent a biased decision
# That can mean anything you want it to. A practical limitation is being able to build a BDD from the map (enumerating all input pairs is quickly infeasible)
def Linf_bdd(bdd, cols, decision, vars_map, distances={}):
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

