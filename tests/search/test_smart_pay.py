# coding: utf-8
from kerasy.search import smart_pay

limits = [0,1,2,4,8]

def test_smart_pay():
    for i,limit in enumerate(sorted(limits)):
        combs = smart_pay(coins=[1000,500,100,50,10,5,1],
                          total=7332,
                          limit=limit,
                          retval=True,
                          verbose=-1)

        num_coins = len(combs)

        assert i==0 or num_coins <= prev_num_coins
        prev_num_coins = num_coins
