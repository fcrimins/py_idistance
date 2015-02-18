from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_numpy_include_dirs

# import numpy as np
# print("HEY HO 1: {}".format(np.get_include()))
# print("HEY HO 2: {}".format(get_numpy_include_dirs()))

# specifying include_path in this manner doesn't have any effect on the generated compiler command
#extensions = cythonize("idist_cython.pyx", include_path=[np.get_include()])

extensions = [
        Extension("*", ["*.pyx"],
                  include_dirs=get_numpy_include_dirs()),
    ]
                  
extensions = cythonize(extensions)

extensions.append(
    Extension("cIdist", sources=["idist.c"],
              include_dirs = get_numpy_include_dirs() # needed to install Visual C++ 2008 Express explained here: http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat
        ))

setup(
    ext_modules = extensions,
)
