import platform
from setuptools import setup

DISTNAME = 'kerasy'
LICENSE = None
DESCRIPTION = 'A set of python modules for machine learning and data mining especially in the biological field.'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Shuto Iwasaki'
MAINTAINER_EMAIL = 'cabernet.rock@gmail.com'
URL = 'https://iwasakishuto.github.io/Kerasy/doc'
DOWNLOAD_URL
PROJECT_URLS = {
    'Documentation': 'https://iwasakishuto.github.io/Kerasy/doc/index.html',
    'Source Code': 'https://github.com/iwasakishuto/Kerasy'
}

VERSION = "0.0.0"

# import kerasy
# VERSION = kerasy.__version__

if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.19.1'
    NUMPY_MIN_VERSION = '1.13.3'

JOBLIB_MIN_VERSION = '0.11'

def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    python_requires=">=3.6",
                    install_requires=[
                        'numpy>={}'.format(NUMPY_MIN_VERSION),
                        'scipy>={}'.format(SCIPY_MIN_VERSION),
                        'joblib>={}'.format(JOBLIB_MIN_VERSION)
                    ])
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
