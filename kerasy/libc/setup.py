# codin: utf-8
# python setup.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("c_kmeans", sources=["c_kmeans.pyx"], include_dirs=['.', get_include()])
setup(name="c_kmeans", ext_modules=cythonize([ext]))
