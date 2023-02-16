# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:49:35 2023

@author: runep
"""
import DFT
import numpy as np
import copy
import pytest

@pytest.fixture(scope = 'session')
def get_DFT_test_data():
    # sampling rate
    sr = 1024
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

@pytest.fixture(autouse = True)
def setup_and_teardown():
    print('\nFetching data from db')
    yield
    print('\nSaving test run data in db')
    
def test_DFT(get_DFT_test_data):
    x = get_DFT_test_data
    
    N = len(x)
    Xr_o = np.zeros((x.shape))
    Xi_o = np.zeros((x.shape))
    xr = copy.deepcopy(x)
    xi = copy.deepcopy(x)
    
    X = np.fft.fft(x)
    X_k, time = DFT.DFT(xr, xi, Xr_o, Xi_o, N)
    
    assert np.allclose( abs(X), abs(X_k[0]) )
    assert np.allclose( abs(X), abs(X_k[1]) )
    