"""
Created on Thu Jan 2 2020
@author: Shuto Iwasaki

```
$ python setup.old.py build_ext --inplace
```

class distutils.core.Extension
Ref: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
    * name   : the full name of the extension, including any packages
               — ie. not a filename or pathname, but Python dotted name.
    * sources: list of source filenames, relative to the distribution root (where
               the setup script lives), in Unix form (slash-separated) for portability.
"""
# codin: utf-8
import os
from pathlib import Path
import numpy as np

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

CLIB_ABS_PATH = os.path.abspath(os.path.dirname(__file__))

def setup_clib():
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    extentions = []
    p = Path(CLIB_ABS_PATH)
    for abs_prog_path in p.glob("*.pyx"):
        fn = abs_prog_path.name # hoge.pyx
        name = fn.split(".")[0] # hoge
        extentions.append(
            Extension(
                name=name,
                sources=[fn],
                include_dirs=[np.get_include()],
                libraries=libraries,
                language="c++"
            )
        )
        print(f"* \033[34m{fn}\033[0m is compiled by Cython to \033[34m{name}.c\033[0m file.")

    setup(
        name="clib",
        cmdclass = {'build_ext': build_ext},
        ext_modules=cythonize(extentions)
    )


if __name__ == "__main__":
    setup_clib()
