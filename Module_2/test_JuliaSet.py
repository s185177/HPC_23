# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:26:28 2023

@author: runep
"""
import pytest
import JuliaSet as js

@pytest.fixture(scope = 'session')
def test_setup_data():
    desired_width = 1000
    max_iterations = 300
    return desired_width, max_iterations 

@pytest.fixture(autouse = True)
def setup_and_teardown():
    print('\nFetching data from db')
    yield
    print('\nSaving test run data in db')

def test_calc_pure_python(test_setup_data):
    val1, val2 = test_setup_data
    output = js.calc_pure_python(val1, val2)
    assert sum(output) == 33219980