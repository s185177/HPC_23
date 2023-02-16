# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:54:31 2023

@author: runep
"""
import pytest
def sum(num1, num2):
    """It returns sum of two numbers"""
    return num1 + num2

@pytest.fixture(scope = 'session')
def get_sum_test_data():
        return [(3,5,8), (-2,-2,-4), (-1,5,4), (3,-5,-2), (0,5,5)]

@pytest.fixture(autouse = True)
def setup_and_teardown():
    print('\nFetching data from db')
    yield
    print('\nSaving test run data in db')
    
# @pytest.mark.parametrize('num1, num2, expected', get_sum_test_data())
def test_sum(get_sum_test_data):
    for data in get_sum_test_data:
        num1 = data[0]
        num2 = data[1]
        expected = data[2]
        assert sum(num1, num2) == expected
    
# def test_sum_output_type():
#     assert type(sum(1, 2)) is int
    
