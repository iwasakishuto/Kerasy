"""
Created on Thu Jan 2 2020
@author: Shuto Iwasaki
```
$ python setup.py build_ext --inplace
```
"""

# codin: utf-8
import os
from pathlib import Path

from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

# ext = Extension("c_kmeans", sources=["c_kmeans.pyx"], include_dirs=['.', get_include()])
# setup(name="c_kmeans", ext_modules=cythonize([ext]))

if __name__ == "__main__":
    extentions = []

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    p = Path()
    for fn in p.glob("*.pyx"):
        fn = str(fn)
        print(f"* \033[34m{fn}\033[0m")
        name = fn.split(".")[0]
        extentions.append(
            Extension(
                name=name,
                sources=[fn],
                include_dirs=['.', get_include()],
                libraries=libraries,
                language="c++"
            )
        )
    setup(name="Kerasy_clib", ext_modules=cythonize(extentions))
