# coding: utf-8
# ref: https://packaging.python.org/tutorials/packaging-projects/

import setuptools

VERSION = "0.0.0"
DESCRIPTION = \
'A set of python modules for machine learning and data mining \
especially in the biological field.'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

def setup_package():
    metadata = dict(
        name='kerasy',
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Shuto Iwasaki',
        author_email='cabernet.rock@gmail.com',
        license='MIT',
        url='https://iwasakishuto.github.io/Kerasy/doc',
        project_urls={
            'Documentation': 'https://iwasakishuto.github.io/Kerasy/doc/index.html',
            'Source Code': 'https://github.com/iwasakishuto/Kerasy',
        },
        packages=setuptools.find_packages(),
        python_requires=">=3.6",
        install_requires=[
            'numpy>=1.15.1',
            'scipy>=1.4.1',
            'seaborn>=0.10.0',
            'Cython>=0.28.5',
        ],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    )
    setuptools.setup(**metadata)

if __name__ == "__main__":
    setup_package()
