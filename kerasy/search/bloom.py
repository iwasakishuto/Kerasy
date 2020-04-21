# coding: utf-8
import numpy as np
import bitarray

from ..utils import handleTypeError
from ..utils import make_hashfuncs

class BloomFilter():
    """
    - M is the filter size. (M = m*k)
    - k is the number of hash functions.
    - P is the False Positive probability. (P = p^k)
    ~~~
    Optimize the parameters.
    The probability that a specific bit in a slice is set after n insertions is
    p  =  1 - (1 - 1/m)^n  ≈  1 - e^(-n/m) (∵ Taylor series expansion)
    n  ≈  -m * ln(1-p)  = M ln(p)ln(1-p) / -ln(P)
    For any given error probability P and filter size M,
    n is maximized by making p=1/2, regardless of P or M.
    As p corresponds to the fill ration of a slice, a filter depicts an optimal use
    when slices are half full. With p=1/2 we obtain
    n  ≈  M * (ln(2))^2 / abs(ln(P))
    →  M  ≈  n * abs(ln(P)) / ((ln(2))^2)
    →  m  ≈  n * abs(ln(P)) / (k * (ln(2))^2)
    k  =  log_2(1/P)
    ~~~
    Reference:
    * P. Almeida, C.Baquero, N. Preguiça, D. Hutchison, "Scalable Bloom Filters".
    """
    def __init__(self, capacity, error_rate=0.001):
        """
        @params capacity   : (int)   n
        @params error_rate : (float) P
        """
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0")
        # k  =  log_2(1/P)
        self.num_slices = max(int(round(np.log2(1/error_rate))), 1)
        # m  ≈  n * abs(ln(P)) / (k * (ln(2))^2)
        self.bits_per_slice = int(round(capacity*abs(np.log(error_rate)) / (self.num_slices * np.log(2)**2)))
        self.num_bits = self.num_slices * self.bits_per_slice
        # P, n (fixed)
        self.error_rate = error_rate
        self.capacity = capacity
        self.make_hashes = make_hashfuncs(
            bits_size=self.num_bits, num_hashes=self.num_slices, method="partition"
        )
        self._init_bitarray()

    def _init_bitarray(self):
        self.count = 0
        self.bitarray = bitarray.bitarray(self.num_bits, endian='little')
        self.bitarray.setall(False)

    @property
    def params(self):
        return (self.capacity, self.error_rate)

    @property
    def bit_fill_ratio(self):
        return self.bitarray.count() / self.bitarray.length()

    def __contains__(self, e):
        return self.has(e)

    def __len__(self):
        return self.count

    def __add__(self, e):
        self.add(e)
        return self

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def copy(self):
        new_bf = BloomFilter(self.capacity, self.error_rate)
        new_bf.bitarray = self.bitarray.copy()
        return new_bf

    def make_hashfuncs(self, num_data):
        # p = M/N * ln2 is the optimal number.
        opt_hashnum = int(max(round(self.capacity/num_data * np.log(2)), 1))
        hashnum = self.hashnum or opt_hashnum

    def add(self, e):
        if self.count > self.capacity:
            raise IndexError("BloomFilter is at capacity.")
        bitarray = self.bitarray
        bits_per_slice = self.bits_per_slice
        hashes = self.make_hashes(e)
        offset = 0
        for k in hashes:
            if not bitarray[offset + k]:
                found_all_bits = False
            self.bitarray[offset + k] = True
            offset += bits_per_slice
        self.count += 1

    def has(self, e):
        bits_per_slice = self.bits_per_slice
        bitarray = self.bitarray
        hashes = self.make_hashes(e)
        offset = 0
        for k in hashes:
            if not bitarray[offset + k]:
                return False
            offset += bits_per_slice
        return True

    def union(self, other):
        if self.params != other.params:
            raise ValueError(
                "Unioning filters requires both filters to have \
                 both the same capacity and error rate"
            )
        new_bf = self.copy()
        new_bf.bitarray = new_bf.bitarray | other.bitarray
        return new_bf

    def intersection(self, other):
        if self.params != other.params:
            raise ValueError(
                "Intersecting filters requires both filters to have \
                 both the same capacity and error rate"
            )
        new_bf = self.copy()
        new_bf.bitarray = new_bf.bitarray & other.bitarray
        return new_bf
