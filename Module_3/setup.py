# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:58:21 2023

@author: runep
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("STREAM_benchmarkfn.pyx",
                            compiler_directives={"language_level": "3"}),
                            include_dirs=[numpy.get_include()])