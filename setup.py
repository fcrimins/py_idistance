from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_numpy_include_dirs

extensions = cythonize("idist_cython.pyx")

extensions.append(
    Extension("cIdist", sources=["idist.c"],
              include_dirs = get_numpy_include_dirs() # needed to install Visual C++ 2008 Express explained here: http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat
        ))

setup(
    ext_modules = extensions,
)
