# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 23:15:33 2023

@author: runep
"""
from timeit import default_timer as timer
import numpy as np
from sys import getsizeof as sizeof
import array

def create_structure(struct_type, size):
    template = np.ones((size), dtype='float')
    
    if struct_type == 'list':
        a = list(template * 1.0)
        b = list(template * 2.0)
        c = list(template * 0.0)
        
    elif struct_type == 'array':
        a = array.array('d', template * 1.0)
        b = array.array('d', template * 2.0)
        c = array.array('d', template * 0.0)
        
    elif struct_type == 'numpy':
        a = template
        b = template * 2.0
        c = template * 0.0
    
    return a, b, c

def benchmarking(a,b,c, STREAM_ARRAY_SIZE):
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

    return times

def benchmarking_numpy(a, b, c, STREAM_ARRAY_SIZE):
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

    return times
#%%
types = ["list", "array", "numpy"]
kernels = ["Copy", "Scale", "Add", "Triad"]
n_samples = 2000
samples = np.linspace(10, 100_000, n_samples, dtype=int)
time_table = np.zeros((len(types),len(kernels),n_samples))
STREAM_ARRAY_TYPES = np.zeros((3))

for i, t in enumerate(types):
    for j, s in enumerate(samples):
        print("Row %i column %s" %(i,j))
        a,b,c = create_structure(t, s)
        STREAM_ARRAY_TYPES[i] = sizeof(a[-1])
        if t != "numpy":
            time_table[i,:,j] = benchmarking(a, b, c, s)
        else:
            time_table[i,:,j] = benchmarking_numpy(a, b, c, s)

bandwidth_table = np.zeros((3,n_samples))
sum_time = np.sum(time_table, axis = 1)

for i in range(len(types)):
    for STREAM_ARRAY_SIZE in range( n_samples):
        bandwidth_table[i,STREAM_ARRAY_SIZE] = (1.0e-09 * 10 * \
        STREAM_ARRAY_TYPES[i] * STREAM_ARRAY_SIZE)             \
                   / sum_time[i,STREAM_ARRAY_SIZE]
        
        # Dependandt on sample sizes which is expected to be correct, 
        # but gives bandwidth around 300 GB/s
        # bandwidth_table[i,STREAM_ARRAY_SIZE] = (1.0e-09 * 10 * \
        # STREAM_ARRAY_TYPES[i] * samples[STREAM_ARRAY_SIZE])    \
        #                    / sum_time[i,STREAM_ARRAY_SIZE]

import matplotlib.pyplot as plt
fs = 18
plt.figure(figsize=(16,10))
for i in range(3):
    plt.plot(samples,bandwidth_table[i], '-', label=types[i], linewidth=2)
# plt.yscale("log")
# plt.xscale("log")
# plt.ylim([1.5e-1, 4e2])
# plt.ylim([0, 0.016])
# plt.xlim([50, samples[-1]])
plt.xlabel("Array Size", fontsize=fs)
plt.ylabel("Bandwidth [GB/s]", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(loc="upper right", fontsize=fs)
plt.grid()
plt.show()
