#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd

import NBApredFuncs as pf

#Load the pre-processed data
data_all_years = pd.read_csv("pre-processedData_n5.csv")

data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
model_data_df = pd.concat([data2015_df, data2016_df])

validation_data_df = data_all_years.loc[data_all_years['Season'] == 5]

#Separate training and testing set
training_df = model_data_df.sample(frac=0.4, random_state=1)
indlist=list(training_df.index.values)

testing_df = model_data_df.copy().drop(index=indlist)

#Set the control switches for the layered model
plot_gains = True
fixed_wager = True
wager_pct = 0.1

#Select the classification features & model hyperparams
#away_features = ['teamFG%','teamEFG%','teamOrtg','teamEDiff']
#home_features = ['opptTS%','opptEFG%','opptPPS','opptDrtg','opptEDiff','opptAST/TO','opptSTL/TO']
away_features = ['teamDayOff','teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
home_features = ['opptDayOff','opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']

class_features = ['teamAbbr','opptAbbr','Season'] + away_features + home_features
output_label = ['teamRslt'] 
#Note teamRslt = Win means visitors win, teamRslt = Loss means home wins

class_params = [300, 0.01, 5] 
classifier_model = 'GB'

#Select the regression features & model hyperparams
reg_features = ['V ML', 'H ML', 'Pred Probs V', 'Pred Probs H']
reg_label = ['Classifier Profit']

reg_params = [300, 5]   
reg_threshold = 0.1
#I think I should leave reg_threshold fixed during the model optimization since the
#testing and validation data behave so differently when it is > 0 vs = 0

#Train and test the layered model
class_model, classifier_feature_importance, profit_reg_model, testing_gains_reg = pf.layered_model_TrainTest(training_df, testing_df, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct)

#Validate the model using the unseen 2017 data
validation_test = True
if validation_test:
    val_gains_reg = pf.layered_model_validate(validation_data_df, class_features, output_label, class_model, reg_features, profit_reg_model, reg_threshold, plot_gains, fixed_wager, wager_pct)

#Given some new games, determine what bets to place
new_bets = False
if new_bets:
    account = 100
    fixed_wager = False
    current_data_df = validation_data_df[0:20]
    bet_placed_indices, wagers = pf.make_new_bets(current_data_df, class_features, output_label, class_model, reg_features, reg_label, profit_reg_model, reg_threshold, fixed_wager, wager_pct, account)
