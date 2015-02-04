from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# 
# import distutils.sysconfig as sysconfig
# for v in ('BINLIBDEST', 'EXE', 'INCLUDEPY', 'LIBDEST', 'SO', 'VERSION', 'exec_prefix', 'prefix'):
#     print('{} = {}'.format(v, sysconfig._config_vars.get(v)))
# #exit()
# for v in ('OPT', 'CCSHARED'): # 'BINLIBDEST', 'EXE', 'INCLUDEPY', 'LIBDEST', 'SO', 'VERSION', 'exec_prefix', 'prefix'):
#     print('{} = {}'.format(v, sysconfig._config_vars.get(v)))
# sysconfig._config_vars['OPT'] = '-fno-strict-aliasing -ggdb -O2 -pipe -fdebug-prefix-map=/home/jt/rel/python-2.7.3-1/python-2.7.3-1/build=/usr/src/debug/python-2.7.3-1 -fdebug-prefix-map=/home/jt/rel/python-2.7.3-1/python-2.7.3-1/src/Python-2.7.3=/usr/src/debug/python-2.7.3-1 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes'
# sysconfig._config_vars['CCSHARED'] = ''
# sysconfig._config_vars['SO'] = '.dll'
# 
# sysconfig._config_vars['BINLIBDEST'] = 'C:/cygwin/lib/python2.7'
# sysconfig._config_vars['EXE'] = '.exe'
# sysconfig._config_vars['INCLUDEPY'] = 'C:/cygwin/usr/include/python2.7'
# sysconfig._config_vars['LIBDEST'] = 'C:/cygwin/lib/python2.7'
# sysconfig._config_vars['VERSION'] = 2.7
# sysconfig._config_vars['exec_prefix'] = 'C:/cygwin'
# sysconfig._config_vars['prefix'] = 'C:/cygwin'
# 
# import os
# for v in ('CC', 'CXX', 'CPP', 'CFLAGS', 'AR', 'ARFLAGS', '', ''):
#     print(os.environ.get(v))
# os.environ["CC"] = 'C:/cygwin/bin/gcc.exe' # "g++-4.7"
# os.environ["CXX"] = 'C:/cygwin/bin/g++.exe' # "g++-4.7"
# os.environ['CFLAGS'] = ''
# os.environ['AR'] = 'C:/cygwin/bin/ar.exe'
# os.environ['ARFLAGS'] = 'rc'
# os.environ['LDSHARED'] = 'C:/cygwin/bin/gcc.exe'
# os.environ['LDFLAGS'] = '-shared -Wl,--enable-auto-image-base -L.'
# 
# import sys
# print(sys.platform)
# sys.platform = 'cygwin'
# print(sys.platform)

# http://stackoverflow.com/questions/16737260/how-to-tell-distutils-to-use-gcc
modules = [Extension("hello",
                     ["hello.pyx"],
                     language = "c",
                     #extra_compile_args=["-fopenmp"],
                     #extra_link_args=["-fopenmp"]
           )]

for e in modules:
    e.cython_directives = {"embedsignature" : True}

setup(name="hello",
     cmdclass={"build_ext": build_ext},
     ext_modules = modules)