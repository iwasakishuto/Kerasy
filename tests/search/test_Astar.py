# coding: utf-8
from kerasy.search.Astar import OptimalTransit

def test_8puzzle():
    n_rows, n_cols = (3,3)
    initial_str = "2,-1,6,1,3,4,7,5,8"
    last_str    = "1,2,3,4,5,6,7,8,-1"
    pannel_lst = OptimalTransit(n_rows=n_rows, n_cols=n_cols,
                                initial_str=initial_str,
                                last_str=last_str,
                                heuristic_method="Manhattan_distance",
                                verbose=-1,
                                retval=True)
