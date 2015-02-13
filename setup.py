from distutils.core import setup, Extension
#from Cython.Build import cythonize

#setup(
    #ext_modules=cythonize("idist.pyx"),
#)

setup(
    ext_modules = [
        Extension("cIdist", sources=["idist.c"]),
    ],
)
