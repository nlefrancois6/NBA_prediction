#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd

import NBApredFuncs as pf

scraped = True
if scraped == False:
    #Load the pre-processed data
    data_all_years = pd.read_csv("Data/pre-processedData_n3_flat.csv")

    data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
    #data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
    model_data_df = pd.concat([data2015_df])

    validation_data_df = data_all_years.loc[data_all_years['Season'] == 5]
else:
    data2015_df = pd.read_csv('Data/pre-processedData_scraped1516_n3.csv')
    data2016_df = pd.read_csv('Data/pre-processedData_scraped1617_n3.csv')
    
    #model_data_df = pd.concat([data2015_df, data2016_df])
    #model_data_df = data_all_years.loc[data_all_years['Season'] == 0]
    
    #validation_data_df = data_all_years.loc[data_all_years['Season'] == 1]
    #validation_data_df = pd.read_csv('Data/pre-processedData_scraped2021_n3.csv')

#Separate training and testing set

#training_df = model_data_df.sample(frac=0.4, random_state=1)
#indlist=list(training_df.index.values)

#testing_df = model_data_df.copy().drop(index=indlist)

training_df = data2015_df
testing_df = data2016_df

#Set the control switches for the layered model
plot_gains = True
fixed_wager = False
wager_pct = 0.1

if scraped == False:
    #Select the classification features & model hyperparams
    #away_features = ['teamFG%','teamEFG%','teamOrtg','teamEDiff']
    #home_features = ['opptTS%','opptEFG%','opptPPS','opptDrtg','opptEDiff','opptAST/TO','opptSTL/TO']
    #away_features = ['teamDayOff','teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
    #home_features = ['opptDayOff','opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']
    
    away_features = ['teamDayOff', 'teamAST', 'teamBLK', 'teamPF', 'teamFGA', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamDRB','teamTREB%','teamDREB%','teamBLKR','teamOrtg','teamDrtg','teamAR']
    home_features = ['opptDayOff','opptAST', 'opptBLK', 'opptPF', 'opptFGA', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptDRB','opptTREB%','opptDREB%','opptBLKR','opptOrtg','opptDrtg','opptAR']

    #'teamAbbr','opptAbbr'
    class_features = away_features + home_features
    output_label = ['teamRslt'] 
    #Note teamRslt = Win means visitors win, teamRslt = Loss means home wins
else:
    away_features = ['away_pace','away_assist_percentage','away_assists','away_block_percentage','away_blocks','away_defensive_rating','away_defensive_rebound_percentage','away_defensive_rebounds','away_effective_field_goal_percentage','away_field_goal_attempts','away_field_goal_percentage','away_field_goals','away_free_throw_attempts','away_free_throw_percentage','away_free_throws','away_offensive_rating','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_three_point_field_goals','away_total_rebound_percentage','away_total_rebounds','away_true_shooting_percentage','away_turnover_percentage','away_turnovers','away_two_point_field_goal_attempts','away_two_point_field_goal_percentage','away_two_point_field_goals']        
    home_features = ['home_pace','home_assist_percentage','home_assists','home_block_percentage','home_blocks','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_effective_field_goal_percentage','home_field_goal_attempts','home_field_goal_percentage','home_field_goals','home_free_throw_attempts','home_free_throw_percentage','home_free_throws','home_offensive_rating','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_three_point_field_goals','home_total_rebound_percentage','home_total_rebounds','home_true_shooting_percentage','home_turnover_percentage','home_turnovers','home_two_point_field_goal_attempts','home_two_point_field_goal_percentage','home_two_point_field_goals']
    
    class_features = away_features + home_features + ['away_score','home_score']
    output_label = ['Winner']


class_params = [300, 0.01, 5] 
classifier_model = 'GB'

#Select the regression features & model hyperparams
reg_features = ['V ML', 'H ML', 'Pred Probs V', 'Pred Probs H']
reg_label = ['Classifier Profit']

reg_params = [300, 5]   
reg_threshold = 0.1

#Train and test the layered model
class_model, classifier_feature_importance, profit_reg_model, testing_gains_reg = pf.layered_model_TrainTest(training_df, testing_df, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct, scrape = scraped)


#Validate the model using the unseen 2017 data
validation_test = False
if validation_test:
    fixed_wager = False
    val_gains_reg = pf.layered_model_validate(validation_data_df, class_features, output_label, class_model, reg_features, profit_reg_model, reg_threshold, plot_gains, fixed_wager, wager_pct)

#Given some new games, determine what bets to place
new_bets = False
if new_bets:
    account = 100
    fixed_wager = False
    current_data_df = validation_data_df
    bet_placed_indices, wagers, recommended_bets = pf.make_new_bets(current_data_df, class_features, output_label, class_model, reg_features, reg_label, profit_reg_model, reg_threshold, fixed_wager, wager_pct, account)
