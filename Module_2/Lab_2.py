# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:15:52 2023

@author: runep
"""


import sys


sys.getsizeof([])
%timeit l = list(range(10_000))

# Python virtual machine allways allocated a global list of integers [-5, 256]
# Reference for int in this range will be same, but if int is outside, new object is created every time
id(-3)


# Increment = peak memory - starting memory
# Peak memory = peak usage during runtime
# Starting memory = usage just before run
%load_ext memory_profiler
%memit [i*i for i in range(100_000)]
%timeit [i*i for i in range(100_000)]


%%memit l = []
for i in range(100_000):
    l.append(i*2)
%%timeit l = []
for i in range(100_000):
    l.append(i*2)


# Tuples are fixed and immutable once declared
t1 = tuple(range(1,6))
t2 = tuple(range(6,11))

t1 + t2


%timeit l = [0,1,2,3,4,5,6,7,8,9]
# Tuples will be faster because it does not add an overhead extra memory
%timeit l = (0,1,2,3,4,5,6,7,8,9)

# NB using tuple() list() and even range() slows it down substantially!



