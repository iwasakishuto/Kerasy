# codin: utf-8
import os
from pathlib import Path
import numpy as np
from Cython.Build import cythonize

CLIB_ABS_PATH = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import get_info
    npymath_info = get_info('npymath')

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(
        package_name='clib',
        parent_name=parent_package,
        top_path=top_path
    )
    p = Path(CLIB_ABS_PATH)
    for abs_prog_path in p.glob("*.pyx"):
        fn = abs_prog_path.name # hoge.pyx
        name = fn.split(".")[0] # hoge
        config.add_extension(
            name=name,
            sources=[fn],
            language="c++",
            # include_dirs=[np.get_include()],
            # libraries=libraries,
            **npymath_info,
        )
        print(f"* \033[34m{fn}\033[0m is compiled by Cython to \033[34m{name}.c\033[0m file.")

    config.ext_modules = cythonize(config.ext_modules)
    # config.add_subpackage('tests')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
