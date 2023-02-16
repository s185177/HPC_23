# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:35:52 2023

@author: runep
"""
from functools import wraps
from timeit import default_timer as timer
import logging
import numpy as np
logging.basicConfig(level=logging.INFO, filename='sample.log')

def timefn(fn):
    """
    Decorator for time measurements

    Parameters
    ----------
    fn : function
        Wraps the measure_time function around chosen function to get \
        function execution time.

    Returns
    -------
    float
        Time difference between function start and finish.

    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result, t2 - t1
    return measure_time

@timefn
# @profile
def DFT(xr, xi, Xr_o, Xi_o, n_elements):
    """
    Calculate Discrete Fourier Transform
    This function is equivalent to the \
    fast-fourier transform from numpy.fft.fft().

    Parameters
    ----------
    xr : array of float64
        Sample array for real part.
    xi : array of float64
        Sample array for imaginary part.
    Xr_o : array of float64
        Array of zeros for DFT real part.
    Xi_o : array of float64
        Array of zeros for DFT imaginary part.
    n_elements : int
        Size of input sample.

    Returns
    -------
    list
        2 x float arrays
        Real- and imaginary part of the DFT of the input sample data
        as a combined array.

    """
    logging.info("Beginning DFT function with input shape {}".format(xi.shape))
    for i in range(n_elements):
        for j in range(n_elements):
            # Real part of X[i]
            Xr_o[i] +=  xr[j]*                     \
            np.cos(j * i * 2*np.pi / n_elements) + \
               xi[j]*np.sin(j * i * 2*np.pi / n_elements)   \
            
            # Imaginary part of X[i]
            Xi_o[i] += -xr[j]*np.sin(j * i * 2*np.pi / n_elements) + \
                        xi[j]*np.cos(j * i * 2*np.pi / n_elements)
    logging.info("DFT done with following results: \n{}".format([Xr_o, Xi_o]))
    return [Xr_o, Xi_o]
