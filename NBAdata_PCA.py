#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:56:48 2021

@author: noahlefrancois
"""
import pandas as pd
#import numpy as np 
#import matplotlib.pyplot as plt 
import sklearn

df = pd.read_csv("pre-processedData_n5.csv")

cols = df.columns.tolist()

cols = cols[0:2] + cols[5:73] + cols[2:3]
df = df[cols]

x_cols = cols[0:-1]
x_df = df[x_cols]

X = df.iloc[:,0:70].values 
y = df.iloc[:,70].values

X_std = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(X), columns = x_df.columns)

pca = sklearn.decomposition.PCA(0.95)
#pca = sklearn.decomposition.PCA(n_components=28)

principalComponents = pca.fit_transform(X_std) 
#The i-th row contains the weights of each feature in the i-th principal component
principalComponents_labelled = pd.DataFrame(pca.components_,columns=X_std.columns)

#To figure out which features are most important, sum each column to find out
#which columns contain the most weight overall
pca_results = principalComponents_labelled.abs().sum()

#Sort the features by weight and get the top N
N = 28
best_features = pca_results.sort_values(ascending=False)[0:28]

