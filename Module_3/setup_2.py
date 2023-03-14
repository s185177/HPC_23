# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:34:05 2023

@author: runep
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("Gauss_Seidel_solverfn.pyx",
                            compiler_directives={"language_level": "3"}),
                            include_dirs=[numpy.get_include()])