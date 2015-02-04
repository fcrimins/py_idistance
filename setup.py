
from distutils.core import setup
from Cython.Build import cythonize

import os
os.environ["CC"] = '/cygdrive/c/cygwin/bin/gcc' # "g++-4.7"
os.environ["CXX"] = '/cygdrive/c/cygwin/bin/g++'

setup(
    ext_modules = cythonize("hello.pyx")
)
