""" For developers (Shuto Iwasaki) """

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
        print(fn)
        name = fn.rstrip(".pyx")
        extentions.append(
            Extension(
                name=name,
                sources=[fn],
                include_dirs=['.', get_include()],
                libraries=libraries,
            )
        )
    setup(name="Kerasy_clib", ext_modules=cythonize(extentions))
