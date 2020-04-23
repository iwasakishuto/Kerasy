from distutils.version import LooseVersion

CYTHON_MIN_VERSION = '0.28.5'

def _check_cython_version():
    message = f'Please install Cython with a version >= {CYTHON_MIN_VERSION} \
                in order to build a kerasy from source.'
    try:
        import Cython
    except ModuleNotFoundError:
        raise ModuleNotFoundError(message)

    if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
        message += f'The current version of Cython is {Cython.__version__} \
                     installed in {Cython.__path__}.'
        raise ValueError(message)
