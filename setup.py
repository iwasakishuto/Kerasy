# coding: utf-8
# ref: https://packaging.python.org/tutorials/packaging-projects/
import os
import sys
import shutil
from distutils.command.clean import clean as Clean
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

# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function
# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools
    extra_setuptools_args = {
        "zip_safe": False,
        "include_package_data" : True,
    }
else:
    extra_setuptools_args = dict()

class CleanCommand(Clean):
    """ Custom clean command to remove build artifacts. """
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk(DISTNAME):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

cmdclass = {'clean': CleanCommand}
if "sdist" in sys.argv:
    import setuptools
    from distutils.command.sdist import sdist
    cmdclass['sdist'] = sdist

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
        cmdclass=cmdclass,
        python_requires=">=3.6",
        install_requires=[
            'numpy>=1.15.1',
            'scipy>=1.4.1',
            'seaborn>=0.10.0',
            f'Cython>={CYTHON_MIN_VERSION}',
            'pydotplus>=2.0.2',
            'bitarray>=0.8.1',
        ],
        extras_require={
          'tests': ['pytest'],
        },
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
        **extra_setuptools_args,
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
