# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:31:15 2023

@author: runep
"""

from timeit import default_timer as timer
from functools import wraps
from sys import stdout
import matplotlib.pyplot as plt
import torch
import numpy as np
import cupy as cp
import Gauss_Seidel_solverfn as gs

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        # print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result, t2 - t1
    return measure_time

# @profile
def gauss_seidel_pure_python(f):
    f_new = f.copy()
    for i in range(1,f_new.shape[0]-1):
        for j in range(1,f_new.shape[1]-1):
            f_new[i,j] = 0.25 * (f_new[i,j+1] + f_new[i,j-1] +
                                   f_new[i+1,j] + f_new[i-1,j])
    return f_new

def poisson_py(grid, port_type):
    grid = port_type.multiply(0.25,                     \
                       port_type.roll(grid,  1, 0) +    \
                       port_type.roll(grid, -1, 0) +    \
                       port_type.roll(grid,  1, 1) +    \
                       port_type.roll(grid, -1, 1))
    grid[0 ,  :] = 0
    grid[-1,  :] = 0
    grid[: ,  0] = 0
    grid[: , -1] = 0
    return grid

def poisson_torch(grid):
    grid = torch.mul(0.25,                              \
                       torch.roll(grid,  1, 0) +        \
                       torch.roll(grid, -1, 0) +        \
                       torch.roll(grid,  1, 1) +        \
                       torch.roll(grid, -1, 1))
    grid[0 ,  :] = 0
    grid[-1,  :] = 0
    grid[: ,  0] = 0
    grid[: , -1] = 0
    return grid
    
def init_numpy(grid_size):
    x = np.random.rand(grid_size, grid_size)
    # Impose the values at the boundary equal to zero
    x[0 ,  :] = 0
    x[-1,  :] = 0
    x[: ,  0] = 0
    x[: , -1] = 0
    return x
 
def run_grid_experiment(n_samples, n_times , plot, test_cython, save):
    samples_size = np.linspace(10, 100, n_samples, dtype=int)
    time_table = np.zeros((len(samples_size), n_times), dtype=np.float64)
    max_iter = 1_000
    for i, N_size in enumerate(samples_size):
        for j in range(n_times):
            x = init_numpy(N_size)
            t_1 = timer()
            for k in range(max_iter):
                stdout.write("\rProgress: Size %s try %s iter %s of %s"     \
                             %(N_size, j, k, max_iter))
                stdout.flush()
                if test_cython:
                    x = gs.gauss_seidel_numpy(x)
                else:
                    x = gauss_seidel_pure_python(x)
            time_table[i, j] = timer() - t_1
    
    avr = np.mean(time_table, axis=1)
    std = np.std( time_table, axis=1)    
    if save:
        if test_cython:
            np.save('data/avr_fn', avr)
            np.save('data/std_fn', std)
        else:
            np.save('data/avr', avr)
            np.save('data/std', std)
            
    if plot:
        fs = 18
        plt.figure(figsize=(16,12))    
        plt.errorbar(samples_size, 
                     avr, 
                     2*std, 
                     color='red', 
                     linestyle='None', 
                     fmt='-', 
                     linewidth=4)
        plt.plot(samples_size, 
                 avr, 
                 label="Pure python", 
                 linewidth=3)
        plt.xlabel("Array size", fontsize = fs)
        plt.ylabel("Time [s]", fontsize = fs)
        plt.xticks(samples_size,fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.grid()
        plt.show()

@timefn
# @profile
def run_profiler_experiment(x):
    for i in range(1_000):
        stdout.write("\rProgress %i of %i \t" %(i, 1_000))
        stdout.flush()
        
        x = gauss_seidel_pure_python(x)

@timefn            
def run_poisson_py_experiment(x, port_type):
    for i in range(1_000):
        x = poisson_py(x, port_type)
    return x

@timefn            
def run_poisson_experiment(x):
    for i in range(1_000):
        x = poisson_torch(x)
    return x


if __name__=="__main__":
    # Part 2.1
    # run_grid_experiment(n_samples=100, n_times=10, 
    #                     plot=False, test_cython=False, save=False)
    # Part 2.2
    # x = init_numpy(100)
    # run_profiler_experiment(x)
    
    # Part 2.4
    # run_grid_experiment(n_samples=10, n_times=10, 
    #                     plot=False, test_cython=True, save=False)    
                        
    # Part 2.5 - Individual PyTorch test
    # x = torch.rand(100,100)
    # x[0 ,  :] = 0
    # x[-1,  :] = 0
    # x[: ,  0] = 0
    # x[: , -1] = 0
    # x_new_torch = run_poisson_experiment(x)
    
    # # Part 2.6 - Individual Cupy test
    # x = cp.random.rand(100, 100)
    # cp.cuda.Stream.null.synchronize()
    # x[0 ,  :] = 0
    # x[-1,  :] = 0
    # x[: ,  0] = 0
    # x[: , -1] = 0
    
    # # # port_type inicates if numpy or cupy is to be used
    # x_new_cupy = run_poisson_py_experiment(x, cp)
    
    # # Part 2.7
    # # Initiate number of iterations to be averaged and sample sizes
    # n_iter = 10
    # n_samples = 30
    # samples = np.linspace(10, 300, n_samples, dtype=int)
    
    # time_torch = np.zeros((n_samples,n_iter), dtype=np.float64)
    # time_cupy  = np.zeros_like(time_torch)
    # time_numpy = np.zeros_like(time_cupy)
    
    # for i, s in enumerate(samples):
    #     stdout.write("\rPyTorch progress %i of %i \t" %(i, n_samples))
    #     stdout.flush()
    #     for j in range(n_iter):
    #         x = torch.rand(s, s)
    #         x[0 ,  :] = 0
    #         x[-1,  :] = 0
    #         x[: ,  0] = 0
    #         x[: , -1] = 0
    #         x_new_torch, time_torch[i,j] = run_poisson_experiment(x)
    # print()
    # for i, s in enumerate(samples):
    #     stdout.write("\rCupy progress %i of %i \t" %(i, n_samples))
    #     stdout.flush()
    #     for j in range(n_iter):
    #         x = cp.random.rand(s, s)
    #         cp.cuda.Stream.null.synchronize()
    #         x[0 ,  :] = 0
    #         x[-1,  :] = 0
    #         x[: ,  0] = 0
    #         x[: , -1] = 0
    #         x_new_cupy, time_cupy[i,j] = run_poisson_py_experiment(x, cp)
    # print()
    # for i, s in enumerate(samples):
    #     stdout.write("\rNumpy progress %i of %i \t" %(i, n_samples))
    #     stdout.flush()
    #     for j in range(n_iter):
    #         x = init_numpy(s)
    #         x_new_numpy, time_numpy[i,j] = run_poisson_py_experiment(x, np)
    
    # avr_torch = np.mean(time_torch, axis=1)
    # std_torch = np.std( time_torch, axis=1)
    # avr_cupy  = np.mean(time_cupy, axis=1)
    # std_cupy  = np.std( time_cupy, axis=1)
    # avr_numpy = np.mean(time_numpy, axis=1)
    # std_numpy = np.std( time_numpy, axis=1)
    
    # fs = 18
    # plt.figure(figsize=(16,12))    
    # plt.errorbar(samples, 
    #               avr_torch, 
    #               std_torch, 
    #               color='red', 
    #               linestyle='None', 
    #               fmt='-', 
    #               linewidth=4)
    # plt.plot(samples, 
    #           avr_torch, 
    #           label="PyTorch", 
    #           linewidth=3)
    # plt.errorbar(samples, 
    #               avr_cupy, 
    #               std_cupy, 
    #               color='red', 
    #               linestyle='None', 
    #               fmt='-', 
    #               linewidth=4)
    # plt.plot(samples, 
    #           avr_cupy, 
    #           label="Cupy", 
    #           linewidth=3)
    # plt.errorbar(samples, 
    #               avr_numpy, 
    #               std_numpy, 
    #               color='red', 
    #               linestyle='None', 
    #               fmt='-', 
    #               linewidth=4)
    # plt.plot(samples, 
    #           avr_numpy, 
    #           label="Numpy", 
    #           linewidth=3)
    
    # # plt.yscale("log")
    # plt.xlabel("Array size", fontsize = fs)
    # plt.ylabel("Time [s]", fontsize = fs)
    # plt.xticks(fontsize=fs)
    # plt.yticks(fontsize=fs)
    # plt.grid()
    # plt.legend(loc="center right", fontsize=fs)
    # plt.show()
    
    # # Part 2.8
    # import h5py
    # f = h5py.File('Module_3_results_1.h5', 'w')
    # f.create_dataset("Numpy_results", data=x_new_numpy)
    # f.create_dataset("Cupy_results", data=x_new_cupy.get())
    # f.create_dataset("Torch_results", data=x_new_torch)
    # print("Following results are saved: {}".format(f.keys()))
