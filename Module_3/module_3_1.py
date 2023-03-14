# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:28:20 2023

@author: runep
"""

from timeit import default_timer as timer
from functools import wraps
from sys import stdout
from sys import getsizeof as sizeof
import numpy as np
import matplotlib.pyplot as plt
import STREAM_benchmarkfn
import STREAM_benchmark


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        # print(f"{t2 - t1}")
        print()
        return result
    return measure_time

@timefn
def run_experiment_numpy(n_samples, largest_size, test_type):
    samples = np.linspace(10, largest_size, n_samples, dtype=int)
    bandwidth = np.zeros(n_samples, dtype=np.float64)
    for i, s in enumerate(samples):
        stdout.write("\rProgress %i of %s \t" %(i, n_samples))
        stdout.flush()
        if test_type != STREAM_benchmark:
            times, STREAM_ARRAY_TYPE = test_type.benchmarking_numpy(s)
        else:
            times, STREAM_ARRAY_TYPE = test_type.benchmarking_numpy(s)
        bandwidth[i] = (1.0e-09 * 10 *                                  \
        sizeof(STREAM_ARRAY_TYPE[0]) * s)                               \
        / np.sum(times)
    return samples, bandwidth

@timefn
def run_experiment_array(n_samples, largest_size, test_type):
    samples = np.linspace(10, largest_size, n_samples, dtype=int)
    bandwidth = np.zeros(n_samples, dtype=np.float64)
    for i, s in enumerate(samples):
        stdout.write("\rProgress %i of %s \t" %(i, n_samples))
        stdout.flush()
        if test_type != STREAM_benchmark:
            times, STREAM_ARRAY_TYPE = test_type.benchmarking_array(s)
        else:
            times, STREAM_ARRAY_TYPE = test_type.benchmarking_array(s)
        bandwidth[i] = (1.0e-09 * 10 *                                  \
        sizeof(STREAM_ARRAY_TYPE[0]) * s)                               \
        / np.sum(times)
    return samples, bandwidth

def plot_figure(x_1, y_1, label_1, x_2, y_2, label_2):
    fs = 18
    plt.figure(figsize=(16,10))
    plt.plot(x_1, y_1, '-', label=label_1, linewidth=2)
    plt.plot(x_2, y_2, '-', label=label_2, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Array Size", fontsize=fs)
    plt.ylabel("Bandwidth [GB/s]", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="upper right", fontsize=fs)
    plt.grid()
    plt.show()
if __name__=="__main__":
    samples_cython_numpy, bandwidth_cython_numpy =                      \
        run_experiment_numpy(50_000,100_000, STREAM_benchmarkfn)
    samples_normal_numpy, bandwidth_normal_numpy =                      \
        run_experiment_numpy(50_000,100_000, STREAM_benchmark)
    
    samples_cython_array, bandwidth_cython_array =                      \
        run_experiment_array(1_000,200_000, STREAM_benchmarkfn)
    samples_normal_array, bandwidth_normal_array =                      \
        run_experiment_array(1_000, 200_000, STREAM_benchmark)
    
    plot_figure(samples_cython_array, bandwidth_cython_array, "Cython",
                samples_normal_array, bandwidth_normal_array, "Normal array")
    plot_figure(samples_cython_numpy, bandwidth_cython_numpy, "Cython",
                samples_normal_numpy, bandwidth_normal_numpy, "Numpy")
