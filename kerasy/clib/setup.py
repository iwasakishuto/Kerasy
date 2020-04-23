# codin: utf-8
import os
from pathlib import Path
import numpy as np

libraries = []
if os.name == 'posix':
    libraries.append('m')

CLIB_ABS_PATH = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(
        package_name='clib',
        parent_name=parent_package,
        top_path=top_path
    )
    p = Path(CLIB_ABS_PATH)
    for abs_prog_path in p.glob("*.pyx"):
        fn = str(abs_prog_path).split("/")[-1]
        name = fn.split(".")[0]
        config.add_extension(
            name=name,
            sources=[fn],
            include_dirs=[np.get_include()],
            language="c++",
        )
    # config.add_subpackage('tests')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
