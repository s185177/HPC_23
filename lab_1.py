# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:15:16 2023

@author: runep
"""

import psutil as ps
import time
from timeit import default_timer as timer
import timeit
import numpy as np
#%% Exercise 2.1

# Method for determining clock granularity, uses python timer as parameter, return granularity
def checktick(time_func):
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  time_func() # get timestamp from timer
        t2 =  time_func() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = time_func() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
        
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    
    return minDelta

# Define timers as methods so they can be used as checktick parameter
def t_f_1():
    return time.time()
def t_f_2():
    return timer()
def t_f_3():
    return time.time_ns()

# Get clock granularities from each python timer and determine the smallest
time_1 = checktick(t_f_1)
time_2 = checktick(t_f_2)
time_3 = checktick(t_f_3)

