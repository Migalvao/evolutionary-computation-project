"""
statistics.py
Descriptive and inferential statistics in Python.
Uses numpy, scipy and matplotlib.
ernesto@dei.uc.pt
"""
__author__ = 'Ernesto Costa'
__date__ = 'March 2022'
# DISCLAIMER: This code was obtained from Evolutionary Computation's practical classes material

import numpy as np
import scipy.stats as st

# Parametric assessement
def test_normal_sw(data):
    """Shapiro-Wilk"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.shapiro(norm_data)

# Non Parametric
def wilcoxon(data1,data2):
    """
    non parametric
    two samples
    dependent
    """     
    return st.wilcoxon(data1,data2)