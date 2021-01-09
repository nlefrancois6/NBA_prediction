#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:46:53 2021

@author: noahlefrancois
"""

import pandas as pd
import matplotlib.pyplot as plt

from ast import literal_eval

gridSearch_results = pd.read_csv('Data/hyperparam_testing_nEstClass_depth3.csv')
#gridSearch_results = pd.read_csv('Data/hyperparam_testing_maxdepthReg.csv')
num_pts = len(gridSearch_results)

params_store = []
c1_store = []
c2_store = []
c3_store = []
r1_store = []
r2_store = []
n1_store = []

test_profit_store = []
val_profit_store = []
for i in range(num_pts):
     params_store.append(literal_eval(gridSearch_results['Params'][i]))
     c1_store.append(params_store[i][0])
     #c2_store.append(params_store[i][1])
     c3_store.append(params_store[i][1])
     r1_store.append(params_store[i][2])
     r2_store.append(params_store[i][3])
     n1_store.append(params_store[i][4])
     
     test_profit_store.append(gridSearch_results['Testing Profit'][i])
     val_profit_store.append(gridSearch_results['Validation Profit'][i])

plt.figure()
plt.plot(c1_store, test_profit_store, label='Testing')
plt.plot(c1_store, val_profit_store, label='Validation')
plt.legend()
plt.xlabel('Max Depth, Classifier')
plt.ylabel('Model Profit')