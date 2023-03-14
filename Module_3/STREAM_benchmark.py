# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 23:15:33 2023

@author: runep
"""
from timeit import default_timer as timer
import numpy as np
import array

def benchmarking_array(STREAM_ARRAY_SIZE):
    temp = np.ones(STREAM_ARRAY_SIZE, dtype=np.float64)
    a = array.array('d', temp)
    b = array.array('d', temp * 2.0)
    c = array.array('d', temp * 0.0)
    
    times = np.zeros((4))
    scalar = (2.0)
    
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

def benchmarking_numpy(STREAM_ARRAY_SIZE):
    a = np.ones( STREAM_ARRAY_SIZE,      dtype=np.float64)
    b = np.full( STREAM_ARRAY_SIZE, 2.0, dtype=np.float64)
    c = np.zeros(STREAM_ARRAY_SIZE,      dtype=np.float64)
    
    times = np.zeros((4))
    scalar = (2.0)
    
    # Copy
    times[0] = timer()
    c = a
    times[0] = timer() - times[0]
    
    # Scale
    times[1] = timer()
    b = scalar*c
    times[1] = timer() - times[1]
    
    # Add
    times[2] = timer()
    c = a + b
    times[2] = timer() - times[2]
    
    # Triad
    times[3] = timer()
    a = b+scalar*c
    times[3] = timer() - times[3]

    return times, a