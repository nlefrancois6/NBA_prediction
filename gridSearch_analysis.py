#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:27:45 2021

@author: noahlefrancois
"""
import pandas as pd
import matplotlib.pyplot as plt

from ast import literal_eval

#gridSearch_results = pd.read_csv('hyperparam_gridSearch_coarse.csv', converters={'Params': eval})
gridSearch_results = pd.read_csv('hyperparam_gridSearch_coarse.csv')

#Get the best performers from the grid search
gridSearch_results_highProfit = gridSearch_results.loc[gridSearch_results['Profit'] > 800]
gridSearch_results_highProfit = gridSearch_results_highProfit.loc[gridSearch_results_highProfit['Profit'] < 1000]
gridSearch_results_highProfit = gridSearch_results_highProfit.reset_index().drop(columns = ['index'])
num_pts = len(gridSearch_results_highProfit)

params_store = []
c1_store = []
c2_store = []
c3_store = []
r1_store = []
r2_store = []
n1_store = []

profit_store = []
for i in range(num_pts):
     params_store.append(literal_eval(gridSearch_results_highProfit['Params'][i]))
     c1_store.append(params_store[i][0])
     c2_store.append(params_store[i][1])
     c3_store.append(params_store[i][2])
     r1_store.append(params_store[i][3])
     r2_store.append(params_store[i][4])
     n1_store.append(params_store[i][5])
     
     profit_store.append(gridSearch_results_highProfit['Profit'][i])

"""
plt.subplot(231)
plt.plot(c1_store, gridSearch_results_highProfit['Profit'],'o')
plt.xlabel('N estimators Class')

plt.subplot(232)
plt.plot(c2_store, gridSearch_results_highProfit['Profit'],'o')
plt.xlabel('Learning rate Class')

plt.subplot(233)
plt.plot(c3_store, gridSearch_results_highProfit['Profit'],'o')
plt.xlabel('Max depth Class')

plt.subplot(234)
plt.plot(r1_store, gridSearch_results_highProfit['Profit'],'o')
plt.xlabel('N estimators Reg')

plt.subplot(235)
plt.plot(r2_store, gridSearch_results_highProfit['Profit'],'o')
plt.xlabel('Max depth Reg')

plt.subplot(236)
plt.plot(n1_store, gridSearch_results_highProfit['Profit'],'o')
plt.xlabel('Num Games Avg')
"""
corr_plots = False
if corr_plots:
    plt.figure()
    plt.subplot(231)
    plt.plot(c1_store, c2_store,'o')

    plt.subplot(232)
    plt.plot(c1_store, c3_store,'o')
    plt.title('C1 correlations')

    plt.subplot(233)
    plt.plot(c1_store, r1_store,'o')

    plt.subplot(234)
    plt.plot(c1_store, r2_store,'o')

    plt.subplot(235)
    plt.plot(c1_store, n1_store,'o')

    plt.figure()
    plt.subplot(231)
    plt.plot(c2_store, c1_store,'o')

    plt.subplot(232)
    plt.plot(c2_store, c3_store,'o')
    plt.title('C2 correlations')

    plt.subplot(233)
    plt.plot(c2_store, r1_store,'o')
    
    plt.subplot(234)
    plt.plot(c2_store, r2_store,'o')

    plt.subplot(235)
    plt.plot(c2_store, n1_store,'o')
    
    plt.figure()
    plt.subplot(231)
    plt.plot(c3_store, c1_store,'o')
    
    plt.subplot(232)
    plt.plot(c3_store, c2_store,'o')
    plt.title('C3 correlations')
    
    plt.subplot(233)
    plt.plot(c3_store, r1_store,'o')
    
    plt.subplot(234)
    plt.plot(c3_store, r2_store,'o')
    
    plt.subplot(235)
    plt.plot(c3_store, n1_store,'o')
    
    plt.figure()
    plt.subplot(231)
    plt.plot(r1_store, c1_store,'o')
    
    plt.subplot(232)
    plt.plot(r1_store, c2_store,'o')
    plt.title('R1 correlations')
    
    plt.subplot(233)
    plt.plot(r1_store, c3_store,'o')
    
    plt.subplot(234)
    plt.plot(r1_store, r2_store,'o')
    
    plt.subplot(235)
    plt.plot(r1_store, n1_store,'o')
    
    plt.figure()
    plt.subplot(231)
    plt.plot(r2_store, c1_store,'o')
    
    plt.subplot(232)
    plt.plot(r2_store, c2_store,'o')
    plt.title('R2 correlations')
    
    plt.subplot(233)
    plt.plot(r2_store, c3_store,'o')
    
    plt.subplot(234)
    plt.plot(r2_store, r1_store,'o')
    
    plt.subplot(235)
    plt.plot(r2_store, n1_store,'o')
    
    plt.figure()
    plt.subplot(231)
    plt.plot(n1_store, c1_store,'o')
    
    plt.subplot(232)
    plt.plot(n1_store, c2_store,'o')
    plt.title('N1 correlations')
    
    plt.subplot(233)
    plt.plot(n1_store, c3_store,'o')
    
    plt.subplot(234)
    plt.plot(n1_store, r1_store,'o')
    
    plt.subplot(235)
    plt.plot(n1_store, r2_store,'o')


