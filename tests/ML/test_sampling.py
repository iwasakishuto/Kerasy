# coding: utf-8
import numpy as np
from kerasy.ML.sampling import GibbsMsphereSampler

def test_gibbs_msphere_sampling(target=0.15):
    radius = 10
    num_samples = 10000
    dimension = 6

    sampler = GibbsMsphereSampler(dimension=dimension, radius=radius)
    sample = sampler.sample(num_samples, verbose=-1)

    norm = np.sum(np.square(sample), axis=1)
    actual = np.count_nonzero(norm <= (radius/2)**2)
    ideal = ((1/2)**dimension) * num_samples

    assert np.all(norm <= radius**2)
    assert abs(actual/ideal-1) <= target
