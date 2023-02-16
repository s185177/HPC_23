# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:54:18 2023

@author: runep
"""

import numpy as np
import copy
import DFT
import matplotlib.pyplot as plt

def get_DFT_test_data(sr):
    # sampling rate
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,1,ts)
    
    freq = 1.
    x = 3*np.sin(2*np.pi*freq*t)
    
    freq = 4
    x += np.sin(2*np.pi*freq*t)
    
    freq = 7   
    x += 0.5* np.sin(2*np.pi*freq*t)
    
    return x

if __name__ == "__main__":
    samples = np.linspace(8,32,4, dtype = int)
    
    logtime_data = np.zeros([len(samples)])
    
    for i, s in enumerate(samples):
        x = get_DFT_test_data(s)
        
        N = len(x)
        Xr_o = np.zeros((x.shape))
        Xi_o = np.zeros((x.shape))
        xr = copy.deepcopy(x)
        xi = copy.deepcopy(x)
    
        X_k, logtime_data[i] = DFT.DFT(xr, xi, Xr_o, Xi_o, N)

    fit = np.polyfit(samples, logtime_data, 2)
    p = np.poly1d(fit)
    
    fs = 18
    
    plt.figure(figsize = (16,10))
    plt.bar(samples, logtime_data, width=10)
    plt.plot(samples,p(samples), '-r')
    plt.xlabel("Array size", fontsize = fs)
    plt.ylabel("Time [s]", fontsize = fs)
    plt.xticks(samples[0::3],fontsize=fs)
    plt.yticks( fontsize=fs)
    plt.ylim([0, 6.5])
    plt.xlim([0, samples[-1]+10])
    plt.savefig('DFT_array_plot.png', dpi=250)
    plt.show()

