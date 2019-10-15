from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import os

from codecs import open

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

#To prepare a new release
#python setup.py sdist upload

setup(name='himatpy',
    version='0.1.0',
    description='Libraries and command-line utilities for HiMAT',
    author='Anthony Arendt',
    author_email='arendta@uw.edu',
    license='MIT',
    url='https://github.com/NASA-Planetary-Science/HiMAT',
    packages=find_packages(),
    long_description=long_description,
    #install_requires=['gdal','numpy','scipy','matplotlib'],
    #Note: this will write to /usr/local/bin
    scripts=['himatpy/modscag_download.py', 'himatpy/LIS/LISpreprocess.py']
)
