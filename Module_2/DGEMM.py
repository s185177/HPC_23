# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:34:42 2023

@author: runep
"""
import random
from timeit import default_timer as timer
from functools import wraps
import numpy as np
import array
import matplotlib.pyplot as plt

def timefn(fn):
    """
    function to measure the time of a function
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds \
              for size {len(result[0])}")
        # print(f"{t2 - t1}")
        return result, t2 - t1
    return measure_time

def generate_list(list_size):
    list_a = [[np.float64(np.random.rand()) for i in range(list_size)] \
              for j in range(list_size)]
    list_b = [[np.float64(np.random.rand()) for i in range(list_size)] \
              for j in range(list_size)]
    list_c = [[np.float64(np.random.rand()) for i in range(list_size)] \
              for j in range(list_size)]
    return [list_a, list_b, list_c]

def generate_array(array_size):
    array_a = [ array.array('d', (np.random.rand()                     \
              for i in range(array_size))) for j in range(array_size)]
    array_b = [ array.array('d', (np.random.rand()                     \
              for i in range(array_size))) for j in range(array_size)]
    array_c = [ array.array('d', (np.random.rand()                     \
              for i in range(array_size))) for j in range(array_size)]
    return [array_a, array_b, array_c]

def generate_numpy(numpy_size):
    return np.float64(np.random.rand(3, numpy_size, numpy_size))

@timefn
def calc_DGEMM(input_struct, struct_label):
    N = len(input_struct[0])
    A = input_struct[0]
    B = input_struct[1]
    C = input_struct[2]
    if struct_label != "numpy":
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C[i][j] = C[i][j] + A[i][k] * B[k][j]
    else:
        C += np.matmul(A,B)
    return C

def calc_execution_time(largest_size, n_samples, n_times, structure_type):
    samples_array = np.linspace(10,largest_size, n_samples, dtype=int)
    time_matrix = np.zeros([n_times, n_samples])
    
    for i in range(n_times):
        for j, s in enumerate(samples_array):
            C, time_matrix[i, j] = calc_DGEMM(generate_list(s), structure_type)

    std = np.std(time_matrix, axis=0)
    avr = np.mean(time_matrix, axis=0)    

    fs = 18
    plt.figure(figsize=(16,12))
    plt.plot(samples_array,avr,'-', linewidth=2, label=structure_type)
    plt.errorbar(samples_array,avr,2*std, linestyle='None', fmt='-o')
    plt.xlabel("Array size", fontsize = fs)
    plt.ylabel("Time [s]", fontsize = fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="center right", fontsize = fs)
    plt.grid()
    plt.savefig(structure_type + ".png", dpi=250)
    plt.show()
    
    return time_matrix
    
if __name__ == "__main__":
    struct_types = ["list", "array", "numpy"]
    
    time_list  = calc_execution_time(400 ,10, 10, struct_types[0])
    time_array = calc_execution_time(400 ,10, 10, struct_types[1])
    time_numpy = calc_execution_time(2000,10, 10, struct_types[2])
    

   
