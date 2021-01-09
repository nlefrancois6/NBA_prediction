#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:28:58 2021

@author: noahlefrancois
"""
import pandas as pd
import NBApredFuncs as pf
import time

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

data2015_df = pd.read_csv('Data/pre-processedData_scraped1516_n3.csv')
data2016_df = pd.read_csv('Data/pre-processedData_scraped1617_n3.csv')
data2017_df = pd.read_csv('Data/pre-processedData_scraped1718_n3.csv')
data2018_df = pd.read_csv('Data/pre-processedData_scraped1819_n3.csv')
data2020_df = pd.read_csv('Data/pre-processedData_scraped2021_n3.csv')
    
#best results w randomized train/test split of 2016+2017, or train 2017 test 2018
training_df = data2017_df
testing_df = data2018_df
validation_data_df = data2016_df

away_features = ['away_pace','away_assist_percentage','away_block_percentage','away_defensive_rating','away_defensive_rebound_percentage','away_field_goal_percentage','away_free_throw_attempts','away_free_throw_percentage','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_total_rebound_percentage','away_turnover_percentage','away_turnovers',]        
home_features = ['home_pace','home_assist_percentage','home_block_percentage','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_field_goal_percentage','home_free_throw_attempts','home_free_throw_percentage','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_total_rebound_percentage','home_turnover_percentage','home_turnovers']
    
    
class_features = away_features + home_features
output_label = ['Winner']

#Select the regression features & model hyperparams
reg_features = ['V ML', 'H ML', 'Pred Probs V', 'Pred Probs H']
reg_label = ['Classifier Profit']

#Set the control switches for the layered model
plot_gains = False
fixed_wager = False
wager_pct = 0.1
wager_crit = 'sqrt'
classifier_model = 'RF'

#Define the hyperparameter search grid
#n_estimators_class_grid = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
n_estimators_class_grid = [80]
#learning_rate_class_grid = [0.05, 0.05, 0.05]
#max_depth_class_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
max_depth_class_grid = [3]

num_classGrid = len(n_estimators_class_grid)*len(max_depth_class_grid)

#n_estimators_reg_grid = [5, 10, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
n_estimators_reg_grid = [100]
#max_depth_reg_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
max_depth_reg_grid = [3]

num_regGrid = len(n_estimators_reg_grid)*len(max_depth_reg_grid)

reg_threshold_grid = 0
n_games_avg = [3]
num_layerGrid = len(n_games_avg)


numGridPts = num_classGrid*num_regGrid*num_layerGrid

#Evaluate the layered model at each point on the grid
params_store = []
test_profit_store = []
val_profit_store = []
#feature_importance_store = []

pts_tested = 0
print('Starting Grid Search')
t1 = time.time()
for c1 in n_estimators_class_grid:
    for c3 in max_depth_class_grid:
        for r1 in n_estimators_reg_grid:
            for r2 in max_depth_reg_grid:
                for n1 in n_games_avg:
                    #for t1 in reg_threshold:
                    
                    pts_tested = pts_tested + 1
                    
                    class_params = [c1, c3] 
                    reg_params = [r1, r2]   
                    reg_threshold = reg_threshold_grid
                    params = [c1, c3, r1, r2, n1, reg_threshold]
                        
                    training_df_n = training_df.loc[training_df['Num Games Avg'] == n1]
                    testing_df_n = testing_df.loc[testing_df['Num Games Avg'] == n1]

                    #Train and test the layered model
                    scraped = True
                    class_model, profit_reg_model, testing_gains_reg = pf.layered_model_TrainTest(training_df, testing_df, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scraped)
                    testing_profit = sum(testing_gains_reg)
                        
                    #Validation of the model
                    val_gains_reg = pf.layered_model_validate(validation_data_df, class_features, output_label, class_model, reg_features, profit_reg_model, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scraped)
                    val_profit = sum(val_gains_reg)

                    #Will want to store the gridpoint params and the profit obtained 
                    #by the gridpoint model on the testing data
                    params_store.append(params)
                    test_profit_store.append(testing_profit)
                    val_profit_store.append(val_profit)
                    
                    #Maybe also store the feature importances, but that would be expensive
                    #feature_importance_store.append(classifier_feature_importance)

                    if pts_tested%10 == 0:
                        print('Grid Point', pts_tested, 'of', numGridPts)

t2 = time.time()
runtime = round((t2-t1)/60,1)
print('Done Grid Search. Run-Time:', runtime,'minutes')

#Store the grid point parameters and profits in a dataframe and save it to a csv
gridSearch_data_dict = {'Testing Profit':test_profit_store, 'Validation Profit': val_profit_store}
gridSearch_data_dict['Params'] = params_store
gridSearch_df = pd.DataFrame(data=gridSearch_data_dict)
gridSearch_df.to_csv('Data/hyperparam_testing_maxdepthReg.csv', index=False)



