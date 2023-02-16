# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:10:21 2023

@author: runep
"""
import pytest
import numpy as np
import DGEMM

@pytest.fixture(scope = 'session')
def get_fixed_dgemm_data():
    c = 0
    n = 3
    test_matrix = np.zeros([3,n,n])

    for i in range(3):
        for j in range(n):
            for k in range(n):
                test_matrix[i,j,k] = c
                c += 1
    expected_matrix = np.array( \
        [[60, 64, 68], [171, 184, 197], [282, 304, 326]], dtype=float)
    return  test_matrix, expected_matrix

@pytest.fixture(scope = 'session')
def get_random_dgemm_data():
    n = 10
    test_matrix = DGEMM.generate_numpy(n)
    expected_matrix = \
        test_matrix[2] + np.matmul(test_matrix[0], test_matrix[1])
    return  test_matrix, expected_matrix

@pytest.fixture(autouse = True)
def setup_and_teardown():
    print('\nFetching data from db')
    yield
    print('\nSaving test run data in db')

def test_fixed_dgemm(get_fixed_dgemm_data):
    input_values, expected_values = get_fixed_dgemm_data
    output_values, time = DGEMM.calc_DGEMM(input_values)
    assert np.allclose(expected_values, output_values)
    assert np.all(np.equal(expected_values, output_values))

def test_random_dgemm_close(get_random_dgemm_data):
    input_values, expected_values = get_random_dgemm_data
    output_values, time = DGEMM.calc_DGEMM(input_values)
    assert np.allclose(expected_values, output_values, atol=1e-14)

def test_random_dgemm_equal(get_random_dgemm_data):
    input_values, expected_values = get_random_dgemm_data
    output_values, time = DGEMM.calc_DGEMM(input_values)
    assert np.all(np.equal(expected_values, output_values))
