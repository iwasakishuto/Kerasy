from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("k_means", sources=["k_means.pyx"], include_dirs=['.', get_include()])
setup(name="k_means", ext_modules=cythonize([ext]))
