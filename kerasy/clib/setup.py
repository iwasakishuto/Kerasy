""" It is necessary to include numpy's C head files. """
# codin: utf-8
import os
from pathlib import Path
import numpy as np

CLIB_ABS_PATH = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import get_info
    npymath_info = get_info('npymath')
    # {
    #     'include_dirs': ['/usr/local/lib/python3.7/site-packages/numpy/core/include'],
    #     'library_dirs': ['/usr/local/lib/python3.7/site-packages/numpy/core/lib'],
    #     'libraries': ['npymath'],
    #     'define_macros': []
    # }

    if os.name == 'posix':
        npymath_info['libraries'].append('m')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(
        package_name='clib',
        parent_name=parent_package,
        top_path=top_path
    )
    p = Path(CLIB_ABS_PATH)
    for abs_prog_path in p.glob("*.pyx"):
        fn = abs_prog_path.name # hoge.pyx
        *name, ext = fn.split(".")
        name = ".".join(name) # hoge
        config.add_extension(
            name=name,
            sources=[fn],
            language="c++",
            **npymath_info,
        )
        print(f"* \033[34m{fn}\033[0m is compiled by Cython to \033[34m{name}.cpp\033[0m file.")

    # config.add_subpackage('tests')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
