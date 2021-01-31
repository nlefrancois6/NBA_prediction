#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:56:48 2021

@author: noahlefrancois
"""
import pandas as pd
#import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


scraped = True
if scraped == False:
    df = pd.read_csv("Data/pre-processedData_n5_flat.csv")
    cols = df.columns.tolist()
    
    cols = cols[0:2] + cols[5:73] + cols[2:3]
    df = df[cols]
    
    x_cols = cols[0:-1]
    x_df = df[x_cols]
    
    X = df.iloc[:,0:70].values 
    y = df.iloc[:,70].values
else:
    df = pd.read_csv("Data/pre-processedData_scraped_inj_1920_n3.csv")
    cols = df.columns.tolist()

    cols = cols[0:68] + cols[71:] + cols[70:71] + cols[68:70]
    df = df[cols]
    
    x_cols = cols[1:-3]
    x_df = df[x_cols]
    
    X = df.iloc[:,1:-3].values 
    y = df.iloc[:,71].values

X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns = x_df.columns)

pca = PCA(0.80)
#pca = sklearn.decomposition.PCA(n_components=28)

#Get the principal components for each game. We could use this as input to the model
principalComponents = pca.fit_transform(X_std) 

#The i-th row contains the weights of each feature in the i-th principal component
principalComponents_labelled = pd.DataFrame(pca.components_,columns=X_std.columns)

#To figure out which features are most important, sum each column to find out
#which columns contain the most weight overall
pca_results = principalComponents_labelled.abs().sum()

#Sort the features by weight and get the top N
N = 33
best_features = pca_results.sort_values(ascending=False)[0:N]

