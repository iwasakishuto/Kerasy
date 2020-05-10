# coding: utf-8
import sys

__copyright__    = "Copyright (C) 2020 Shuto Iwasaki"
__version__      = "0.5.0"

__license__      = "MIT"
__author__       = "Shuto Iwasaki"
__author_email__ = "cabernet.rock@gmail.com"
__url__          = "https://iwasakishuto.github.io/Kerasy/doc"

try:
    __KERASY_SETUP__
except NameError:
    __KERASY_SETUP__ = False


if __KERASY_SETUP__:
    sys.stderr.write('Partial import of kerasy during the build process.\n')
else:
    from . import Bio
    from . import clib
    from . import datasets
    from . import engine
    from . import layers
    from . import ML
    from . import search
