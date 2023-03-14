# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:37:17 2023

@author: runep
"""
import numpy as np
cimport numpy as np

import array
from timeit import default_timer as timer
# from sys import getsizeof as sizeof


#cython: boundscheck=False
def benchmarking_numpy(unsigned int STREAM_ARRAY_SIZE):
    cdef double [:] a = np.ones( STREAM_ARRAY_SIZE,      dtype=np.float64)
    cdef double [:] b = np.full( STREAM_ARRAY_SIZE, 2.0, dtype=np.float64)
    cdef double [:] c = np.zeros(STREAM_ARRAY_SIZE,      dtype=np.float64)
    cdef double[:] times = np.empty(4, dtype=np.float64)
    cdef double scalar = 2.0
    
    # Copy
    times[0] = timer()
    c = np.copy(a)
    times[0] = timer() - times[0]
    
    # Scale
    times[1] = timer()
    b = np.multiply(scalar,c)
    times[1] = timer() - times[1]
    
    # Add
    times[2] = timer()
    c = np.add(a, b)
    times[2] = timer() - times[2]
    
    # Triad
    times[3] = timer()
    a = np.add(b, np.multiply(scalar,c))
    times[3] = timer() - times[3]

    return times, a

#cython: boundscheck=False
def benchmarking_array(unsigned int STREAM_ARRAY_SIZE):
    cdef double[:] a = array.array('d', np.ones( STREAM_ARRAY_SIZE,     dtype=np.float64))
    cdef double[:] b = array.array('d', np.full( STREAM_ARRAY_SIZE, 2.0, dtype=np.float64))
    cdef double[:] c = array.array('d', np.zeros(STREAM_ARRAY_SIZE,      dtype=np.float64))
    cdef double[:] times = np.empty(4, dtype=np.float64)
    cdef double scalar = 2.0
    cdef unsigned int j
    
    # Copy
    times[0] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j]
    times[0] = timer() - times[0]
    
    # Scale
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
     b[j] = scalar*c[j]
    times[1] = timer() - times[1]
    
    # Add
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
     c[j] = a[j]+b[j]
    times[2] = timer() - times[2]
    
    # Triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j]+scalar*c[j]
    times[3] = timer() - times[3]

    return times, a
    