# coding: utf-8
"""
This file contains the following algorithms:
- Longest Common Prefix(LCP)
"""
import numpy as np

from ..utils import inverse_arr

def lcp(string,i,j):
    """ Longest-Common-Prefix """
    l=0; n = len(string)
    while (l+max(i,j)<n) and (string[i+l]==string[j+l]): l+=1
    return l

def LCP_create(string, SA):
    """ Make Auxiliary data structure Height(i) which is for LCParray.
    @params string:
    @params SA    : Suffix Array.
    """
    checkString(string)
    length = len(SA)
    ISA = inverse_arr(SA)
    height = np.zeros(shape=length, dtype=int)
    h = 0
    for i in range(length):
        j = ISA[i]
        if j == length-1:
            height[j] = -1
            h = 0
            continue
        k = SA[j+1]
        h = h-1+lcp(string, i+h-1, k+h-1) if h>0 else lcp(string, i, k)
        height[j] = h
    return height
