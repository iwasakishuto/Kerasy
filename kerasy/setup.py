# codin: utf-8
import os
import sys
import numpy as np
from Cython.Build import cythonize

libraries = []
if os.name == 'posix':
    libraries.append('m')

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(
        package_name='kerasy',
        parent_name=parent_package,
        top_path=top_path
    )

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('Bio')
    config.add_subpackage('clib')
    config.add_subpackage('datasets')
    config.add_subpackage('engine')
    config.add_subpackage('layers')
    config.add_subpackage('ML')
    config.add_subpackage('search')
    config.add_subpackage('utils')
    
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
