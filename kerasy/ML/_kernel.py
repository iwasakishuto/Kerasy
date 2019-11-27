""" Ref: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/ """
#coding: utf-8
import numpy as np

def kernel_handler(inputs, **params):
    if type(inputs) == str:
        try:
            # Overwrite default kwargs.
            func = kernel_dict[inputs]
            for k,v in params.items():
                func.__kwdefaults__[k] = v
            return func
        except KeyError:
            print("Please specify from followings:\n")
            print("\n".join(kernel_dict.keys()))
    else:
        return inputs

def linear_kernel(x, x_prime,*,c=0):
    return x.T.dot(x_prime) + c

def polynomial_kernle(x, x_prime,*,alpha=1, c=0, d=3):
    return (alpha*x.T.dot(x_prime) + c)**d

def gaussian_kernel(x, x_prime,*,sigma=1):
    return np.exp(-sum((x-x_prime)**2)/(2*sigma**2))

def exponential_kernel(x, x_prime,*,sigma=0.1):
    return np.exp(-sum(abs(x-x_prime))/(2*sigma**2))

def laplacian_kernel(x, x_prime,*,sigma=0.1):
    return np.exp(-sum(abs(x-x_prime))/sigma)

def hyperbolic_tangent_kernel(x, x_prime,*,alpha=1,c=0):
    return np.tanh(alpha*x.T.dot(x_prime) + c)

def rational_quadratic_kernel(x, x_prime,*,c=1):
    return 1 - sum((x-x_prime)**2)/(sum((x-x_prime)**2)+c)

def multiquadric_kernel(x, x_prime,*,c=1):
    return np.sqrt(sum((x-x_prime)**2) + c)

def inverse_multiquadric_kernel(x, x_prime,*,c=1):
    return 1/multiquadric_kernel(x,x_prime,c=c)

def log_kernel(x, x_prime,*,d=3):
    return -np.log(sum(abs(x-x_prime)**d) + 1)

kernel_dict = {
    "linear": linear_kernel,
    "polynomial": polynomial_kernle,
    "gaussian": gaussian_kernel,
    "exponential": exponential_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": hyperbolic_tangent_kernel,
    "rational_quadratic": rational_quadratic_kernel,
    "multiquadric": multiquadric_kernel,
    "inverse_multiquadric": inverse_multiquadric_kernel,
    "log": log_kernel,
}
