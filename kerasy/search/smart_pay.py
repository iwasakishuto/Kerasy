# coding: utf-8
from ..utils import flush_progress_bar

def breakdown(combs):
    """ display breakdowns """
    use_coins = sorted(set(combs))
    num_coins = [combs.count(coin) for coin in use_coins]
    total_pay = [n*coin for n,coin in zip(use_coins,num_coins)]
    width_coin  = max([len(str(e)) for e in use_coins]+[len("coins")])
    width_num   = max([len(str(e)) for e in num_coins]+[len("number")])
    width_total = max([len(str(e)) for e in total_pay]+[len("pay"),len(str(sum(total_pay)))])
    width_line  = width_coin+width_num+width_total+2

    print_func = lambda c,n,p: print(f"{c:^{width_coin}}|{n:>{width_num}}|{p:>{width_total}}")
    print_func('coins','number','pay')
    print("="*width_line)
    for coin,num,t in zip(use_coins,num_coins,total_pay):
        print_func(coin,num,t)
    print("-"*width_line)
    print_func('total',sum(num_coins),sum(total_pay))

def smart_pay(coins, total, limit=None, verbose=1):
    """
    Find the minimum number of coin combinations by using Dynamic Programming.
    @params coins: (int list) Coins.
    @params total: (int) Amount of Payment.
    @params limit: (int) Maximum number of times a restricted coin can be used.
    """
    total += 1 # because 0-origin.
    if len(set(coins)) < len(coins):
        raise ValueError("All elements of `coins` must be different integers.")
    restricted = coins[0]
    free_coins = coins[1:]

    if limit is None:
        limit = total//restricted+1
    elif verbose:
        print(f'{restricted} coin can only be used up to {limit} times at the same time.')

    # Initialization.
    B = [0 for _ in range(total)] # Memory for Traceback.
    m = [0 if t==0 else 1 if t in free_coins else float('inf') for t in range(total)]

    # Recursion
    for t in range(1,total):
        cands = [m[t-coin] if (t-coin)>=0 else float('inf') for coin in free_coins]
        if not sum([e!=float('inf') for e in cands])==0:
            minnum = min(cands)
            m[t],B[t] = [(e+1,t-coin) for e,coin in zip(cands,free_coins) if e==minnum][0]
        flush_progress_bar(t-1, total-1, metrics={"minimum": m[t]}, verbose=verbose)

    ms = [(l,m[-1-restricted*l]+l) for l in range(limit+1) if restricted*l<=total]
    num_restricted, num_total = min(ms, key=lambda x:x[1])
    idx = total-1-restricted*num_restricted
    combs = [restricted for _ in range(num_restricted)]
    while idx:
        last = B[idx]
        combs.append(idx-last)
        idx = last
    if verbose: breakdown(combs)
