# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:30:35 2023

@author: runep
"""
import numpy as np
cimport numpy as np

#cython: boundscheck=False
def gauss_seidel_numpy(double[:,:] f):
    cdef double[:,:] f_new = f.copy()
    cdef double scalar = 0.25
    cdef unsigned int i, j

    for i in range(1,f_new.shape[0]-1):
        for j in range(1,f_new.shape[1]-1):
            f_new[i,j] = scalar * (f_new[i,j+1] + f_new[i,j-1] +
                                   f_new[i+1,j] + f_new[i-1,j])
    return f_new