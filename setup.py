from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_numpy_include_dirs

# import numpy as np
# print("HEY HO 1: {}".format(np.get_include()))
# print("HEY HO 2: {}".format(get_numpy_include_dirs()))
# print("HEY HO 3: {}".format(get_numpy_include_dirs() + [r'C:\Users\Michelle\Documents\eclipse_code\scikit-learn\sklearn\neighbors']))

# specifying include_path in this manner doesn't have any effect on the generated compiler command
#extensions = cythonize("idist_cython.pyx", include_path=[np.get_include()])

extensions = [
        Extension("*", ["*.pyx"],
                  include_dirs=get_numpy_include_dirs(),
                  #extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"]
        ),
    ]
                  
#extensions = cythonize(extensions)#, include_path=['.', r'C:\Users\Michelle\Documents\eclipse_code\scikit-learn\sklearn\neighbors'])
# this looks for a pxd file, but my install of sklearn only has a pyd (i.e. a dll) file, it
# doesn't have the source files
extensions = cythonize(extensions)#, include_path=['.', r'C:\Python27\Lib\site-packages'])

# attempt at writing a C extension from scratch without Cython (i.e. difficult)
extensions.append(
    Extension("cIdist", sources=["idist.c"],
              include_dirs = get_numpy_include_dirs() # needed to install Visual C++ 2008 Express explained here: http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat
        ))

setup(
    name='py_idistance',
    version='0.0.1',
    packages=[''],
    url='',
    license='',
    author='quant',
    author_email='',
    description='',

    ext_modules = extensions,
)
