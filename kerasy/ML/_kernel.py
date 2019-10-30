""" Ref: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/ """
#coding: utf-8
import numpy as np

def kernel_handler(inputs):
    if type(inputs) == str:
        try:
            return kernel_dict[inputs]
        except KeyError:
            print("Please specify from followings:\n")
            print("\n".join(kernel_dict.keys()))
    else:
        return inputs

def linear_kernel(x, x_prime, c=1):
    return x.T.dot(x_prime) + c

def polynomial_kernle(x, x_prime, alpha=1, c=0, d=3):
    return (alpha*x.T.dot(x_prime) + c)**d

def gaussian_kernel(x, x_prime, sigma=0.1):
    return np.exp(-sum((x-x_prime)**2)/(2*sigma**2))

def exponential_kernel(x, x_prime, sigma=0.1):
    return np.exp(-sum(abs(x-x_prime))/(2*sigma**2))

def laplacian_kernel(x, x_prime, sigma=0.1):
    return np.exp(-sum(abs(x-x_prime))/sigma)

kernel_dict = {
    "linear": linear_kernel,
    "polynomial": polynomial_kernle,
    "gaussian": gaussian_kernel,
    "exponential": exponential_kernel,
    "laplacian": laplacian_kernel
}
