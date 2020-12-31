#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd
from sklearn import ensemble, metrics
import matplotlib.pyplot as plt

import NBApredFuncs as pf

#Load the pre-processed data
data_all_years = pd.read_csv("pre-processedData_n5.csv")

data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
model_data_df = pd.concat([data2015_df, data2016_df])

validation_data_df = data_all_years.loc[data_all_years['Season'] == 5]

#Select the model features
#away_features = ['teamFG%','teamEFG%','teamOrtg','teamEDiff']
#home_features = ['opptTS%','opptEFG%','opptPPS','opptDrtg','opptEDiff','opptAST/TO','opptSTL/TO']
away_features = ['teamDayOff','teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
home_features = ['opptDayOff','opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']

features = ['teamAbbr','opptAbbr','Season'] + away_features + home_features
output_label = ['teamRslt'] 
#Note teamRslt = Win means visitors win, teamRslt = Loss means home wins

#Separate training and testing set
training_df = model_data_df.sample(frac=0.5, random_state=1)
indlist=list(training_df.index.values)

testing_df = model_data_df.copy().drop(index=indlist)

#Define the features (input) and label (prediction output) for training set
training_features = training_df[features]
training_label = training_df[output_label]
training_odds_v = training_df['V ML']
training_odds_h = training_df['H ML']

#Define features and label for testing set
testing_features = testing_df[features]
testing_label = testing_df[output_label]
testing_odds_v = testing_df['V ML']
testing_odds_h = testing_df['H ML']

#Train the classifier on the first 1/2 of the data

#Train a Gradient Boosting Machine on the data, predict the outcomes, and evaluate accuracy
n_estimators = 100
learning_rate = 0.02
max_depth = 5
gbc, predGB, pred_probsGB, accuracyGB = pf.gbcModel(training_features, training_label, testing_label, testing_features, n_estimators, learning_rate, max_depth)

#Train a Random Forest Classifier on the data, predict the outcomes, and evaluate accuracy
n_estimators = 100
max_depth = 1
random_state = 1
rfc, predRF, pred_probsRF, accuracyRF = pf.rfcModel(training_features, training_label, testing_label, testing_features, n_estimators, random_state, max_depth)

#Feature importance plots
plot_features = False
gb_feature_importance = pf.model_feature_importances(plot_features, gbc, features)
    
    
#Switches to calculate the profit, to use dummy odds, and to set an expected gain betting threshold
dummy_odds = False
min_exp_gain = False
plot_gains = False
wager_pct = 0.1

running_accountGB, testing_gainsGB = pf.evaluate_model_profit(predGB, pred_probsGB, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, plot_gains, dummy_odds = False)
if plot_gains:
    plt.title('GB Model Profit')
    
running_accountRF, testing_gainsRF = pf.evaluate_model_profit(predRF, pred_probsRF, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, plot_gains, dummy_odds = False)
if plot_gains:
    plt.title('RF Model Profit')
    
    #could calculate the actual profit for each game based on the classifier predictions 
    #and store these in an array. Then, train a regression model to predict this profit
    #based on the model confidence and the odds


d = {'Classifier Profit': testing_gainsGB, 'Pred Probs V': pred_probsGB[:,0], 'Pred Probs H': pred_probsGB[:,1], 'Prediction': predGB}
reg_df = pd.DataFrame(data=d)

reg_df['V ML'] = testing_odds_v.reset_index().drop(columns = ['index'])
reg_df['H ML'] = testing_odds_h.reset_index().drop(columns = ['index'])

reg_features = ['V ML', 'H ML', 'Pred Probs V', 'Pred Probs H']
reg_label = ['Classifier Profit']

#Separate the 2nd 1/2 of the data into training and testing data (I might end up using another year of data for testing instead)
training_reg_df = reg_df.sample(frac=0.5, random_state=1)
indlist_reg=list(training_reg_df.index.values)

testing_reg_df = reg_df.copy().drop(index=indlist_reg)

#Define the features and label for training set
training_reg_features = training_reg_df[reg_features]
training_reg_label = training_reg_df[reg_label]

#Define features and label for testing set
testing_reg_features = testing_reg_df[reg_features]
testing_reg_label = testing_reg_df[reg_label]

#Create and train a regression model to predict the profit based on odds and classifier confidence
n_estimators = 100
max_depth = 2
random_state = 1
profit_reg = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
#Train the model on the third 1/4 of the data
profit_reg.fit(training_reg_features, training_reg_label)
#Get the expected profit on the remaining 1/4 of the data
expected_profit_testing = profit_reg.predict(testing_reg_features)

#Do some dumb data formatting to get things in arrays even though alot of this doesn't get
#used when we use the regression threshold instead of expected value
preds_testing = testing_reg_df['Prediction'].reset_index().drop(columns = ['index'])
pred_probs_testing_v = testing_reg_df['Pred Probs V'].reset_index().drop(columns = ['index'])
pred_probs_testing_h = testing_reg_df['Pred Probs H'].reset_index().drop(columns = ['index'])

preds_testing_arr = []
pred_probs_testing = []
for i in range(len(preds_testing)):
    preds_testing_arr.append(preds_testing['Prediction'][i])
    pred_probs_testing.append([pred_probs_testing_v['Pred Probs V'][i], pred_probs_testing_h['Pred Probs H'][i]])

testing_reg_odds_v = testing_reg_df['V ML'].reset_index().drop(columns = ['index'])
testing_reg_odds_h = testing_reg_df['H ML'].reset_index().drop(columns = ['index'])

#Evaluate the layered model profit on the remaining 1/4 of the testing data
#Calculate the profit when we only bet on games the regression expectation favours 
reg_threshold = 1 #Only bet on games with regression expectation > threshold
plot_gains = True

running_account_reg, testing_gains_reg = pf.evaluate_model_profit(preds_testing_arr, pred_probs_testing, testing_label, testing_reg_odds_v, testing_reg_odds_h, min_exp_gain, wager_pct, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_testing)
if plot_gains:
    plt.title('Classification-Regression Layered Model Profit, Testing Data')
    
#Validate the model using the unseen 2017 data
validation_test = True
if validation_test:
    #Format the validation data
    validation_features = validation_data_df[features]
    validation_label = validation_data_df[output_label]
    validation_odds_v = validation_data_df['V ML']
    validation_odds_h = validation_data_df['H ML']
    
    #Get classifier predictions on validation data & evaluate the gains
    predGB_val = gbc.predict(validation_features) 
    pred_probsGB_val = gbc.predict_proba(validation_features)
    
    running_accountGB_val, gainsGB_val = pf.evaluate_model_profit(predGB_val, pred_probsGB_val, validation_label, validation_odds_v, validation_odds_h, min_exp_gain, wager_pct, plot_gains, dummy_odds = False)
    if plot_gains:
        plt.title('Classifier Model Profit, Validation Data')
    
    #Data formatting for the regression
    d_val = {'Classifier Profit': gainsGB_val, 'Pred Probs V': pred_probsGB_val[:,0], 'Pred Probs H': pred_probsGB_val[:,1], 'Prediction': predGB_val}
    reg_val_df = pd.DataFrame(data=d_val)

    reg_val_df['V ML'] = validation_odds_v.reset_index().drop(columns = ['index'])
    reg_val_df['H ML'] = validation_odds_h.reset_index().drop(columns = ['index'])
    
    reg_val_features = reg_val_df[reg_features]
    reg_val_label = reg_val_df[reg_label]
    
    #Get regression predictions on validation data
    expected_profit_val = profit_reg.predict(reg_val_features)
    
    #Data formatting for layered model profit evaluation
    preds_val = reg_val_df['Prediction'].reset_index().drop(columns = ['index'])
    pred_probs_val_v = reg_val_df['Pred Probs V'].reset_index().drop(columns = ['index'])
    pred_probs_val_h = reg_val_df['Pred Probs H'].reset_index().drop(columns = ['index'])

    preds_val_arr = []
    pred_probs_val = []
    for i in range(len(preds_val)):
        preds_val_arr.append(preds_val['Prediction'][i])
        pred_probs_val.append([pred_probs_val_v['Pred Probs V'][i], pred_probs_val_h['Pred Probs H'][i]])

    val_reg_odds_v = reg_val_df['V ML'].reset_index().drop(columns = ['index'])
    val_reg_odds_h = reg_val_df['H ML'].reset_index().drop(columns = ['index'])
    
    running_account_reg_val, val_gains_reg = pf.evaluate_model_profit(preds_val_arr, pred_probs_val, validation_label, val_reg_odds_v, val_reg_odds_h, min_exp_gain, wager_pct, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_val)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Validation Data')
    