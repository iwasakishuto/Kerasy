# coding: utf-8
# ref: https://packaging.python.org/tutorials/packaging-projects/
import os
import sys
import shutil
import builtins
builtins.__KERASY_SETUP__ = True

import kerasy
from kerasy.clib import CYTHON_MIN_VERSION

DISTNAME = 'kerasy'
DESCRIPTION = 'A set of python modules for machine learning and data mining \
especially in the biological field.'

cwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cwd, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    from kerasy.clib import _check_cython_version

    config = Configuration(
        package_name=None,
        parent_name=parent_package,
        top_path=top_path
    )
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True
    )

    _check_cython_version()

    config.add_subpackage('kerasy')
    config.add_data_files(('kerasy', 'LICENSE'))

    return config

def setup_package():
    metadata = dict(
        name=DISTNAME,
        version=kerasy.__version__,
        author=kerasy.__author__,
        author_email=kerasy.__author_email__,
        license=kerasy.__license__,
        url=kerasy.__url__,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        # long_description_content_type='text/markdown',
        project_urls={
            'Bug Reports'  : 'https://github.com/iwasakishuto/Kerasy/issues',
            'Documentation': 'https://iwasakishuto.github.io/Kerasy/doc/index.html',
            'Source Code'  : 'https://github.com/iwasakishuto/Kerasy',
            'Say Thanks!'  : 'https://iwasakishuto.github.io/',
        },
        python_requires=">=3.6",
        install_requires=[
            'numpy>=1.15.1',
            'scipy>=1.4.1',
            'seaborn>=0.10.0',
            f'Cython>={CYTHON_MIN_VERSION}',
            'pydotplus>=2.0.2',
            'bitarray>=0.8.1',
        ],
        package_data={'': ['*.pxd']},
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',

            # Pick your license as you wish
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
        ],
    )

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    'dist_info',
                                                    '--version',
                                                    'clean'))):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)

if __name__ == "__main__":
    setup_package()
