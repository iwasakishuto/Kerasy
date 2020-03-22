# coding: utf-8
"""
'factorization' of a string S is a decomposition of S into factors (S = f1...fn)
based on a certain rule.
"""

def LPF_on_line(SA, LCP):
    """
    Ref: "Computing the Longest Previous Factor"
         (https://doi.org/10.1016/j.ejc.2012.07.011)
    ~~~~~
    @params SA  : Suffix Array.
    @params LCP : Longest Common Prefix array.
    @params n   : String length.
    @return LPF : Longest Previous Factor.
    ~~~~~
    [example] string = "abaabababbabbb"

     r SA Suffix           LCP LPF[SA[r]]
    -------------------------------------
     0  2 aabababbabbb      0       1
     1  0 abaabababbabbb    1       0
     2  3 abababbabbb       3       3
     3  5 ababbabbb         4       4
     4  7 abbabbb           2       2
     5 10 abbb              3       3
     6 13 b                 0       1
     7  1 baabababbabbb     1       0
     8  4 bababbabbb        2       2
     9  6 babbabbb          3       3
    10  9 babbb             4       4
    11 12 bb                1       2
    12  8 bbabbb            2       1
    13 11 bbb               2       2
    -------------------------------------
    """
    n = len(LCP)
    S = [] # Empty Stack
    LPF = np.full(fill_value=-1, shape=(n), dtype=int)

    for r in range(n):
        r_lcp = LCP[r]
        while len(S)>0:
            t,t_lcp = S[-1] # Top(S)
            if SA[r] < SA[t]:
                LPF[SA[t]] = max(t_lcp, r_lcp)
                r_lcp = min(t_lcp, r_lcp)
                S.pop()
            elif (SA[r] > SA[t]) and (r_lcp<=t_lcp):
                LPF[SA[t]] = t_lcp
                S.pop()
            else:
                break
        S.append((r, r_lcp))
    while len(S)>0:
        t,t_lcp = S.pop()
        LPF[SA[t]] = t_lcp
    if np.any(LPF==-1):
        if len(np.unique(SA)) != len(SA):
            raise ValueError("All elements of the Suffix Array must be different!!")
        else:
            raise ValueError("Some element of the LPF array are calculated.")
    return LPF
