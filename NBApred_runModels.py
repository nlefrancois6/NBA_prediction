#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import NBApredFuncs as pf


hist = False
scraped = True
if scraped == False:
    #Load the pre-processed data
    data_all_years = pd.read_csv("Data/pre-processedData_n3_flat.csv")

    data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
    #data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
    model_data_df = pd.concat([data2015_df])

    validation_df = data_all_years.loc[data_all_years['Season'] == 5]
else:
    data2015_df = pd.read_csv('Data/pre-processedData_scraped_inj_1516_n3.csv')
    data2016_df = pd.read_csv('Data/pre-processedData_scraped_inj_1617_n3.csv')
    data2017_df = pd.read_csv('Data/pre-processedData_scraped_inj_1718_n3.csv')
    data2018_df = pd.read_csv('Data/pre-processedData_scraped_inj_1819_n3.csv')
    data2019_df = pd.read_csv('Data/pre-processedData_scraped_inj_1920_n3.csv')
    data2020_df = pd.read_csv('Data/pre-processedData_scraped2021_n3_inj.csv')
    
    #best results w randomized train/test split of 2016+2017, or train 2017 test 2018
    #training_df = pd.concat([data2016_df, data2017_df])
    
    #training_df = data2018_df
    #testing_df = data2019_df
    #validation_df = data2020_df
    
    #half2018 = int(len(data2018_df)/2)
    #training_df = pd.concat([data2017_df, data2018_df[:half2018]])
    #testing_df = pd.concat([data2018_df[half2018:], data2019_df])
    #training_df = pd.concat([data2015_df, data2016_df[:half2016]])
    #testing_df = data2016_df[half2016:]
    
    
    #model_data_df = pd.concat([data2017_df, data2016_df])
    #validation_df = data2020_df
    
    #Don't delete these settings, they give the best results so far
    training_df = data2018_df
    testing_df = pd.concat([data2017_df, data2019_df])
    validation_df = data2020_df


#Separate training and testing set

#training_df = model_data_df.sample(frac=0.4, random_state=1)
#indlist=list(training_df.index.values)

#testing_df = model_data_df.copy().drop(index=indlist)


#Set the control switches for the layered model
plot_gains = True
fixed_wager = False
wager_pct = 0.1
wager_crit = 'sqrt' #options are 'sqrt', 'kelly'

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
    #away_features = ['away_pace','away_assist_percentage','away_assists','away_block_percentage','away_blocks','away_defensive_rating','away_defensive_rebound_percentage','away_defensive_rebounds','away_effective_field_goal_percentage','away_field_goal_attempts','away_field_goal_percentage','away_field_goals','away_free_throw_attempts','away_free_throw_percentage','away_free_throws','away_offensive_rating','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_three_point_field_goals','away_total_rebound_percentage','away_total_rebounds','away_true_shooting_percentage','away_turnover_percentage','away_turnovers','away_two_point_field_goal_attempts','away_two_point_field_goal_percentage','away_two_point_field_goals','away_minutes_played_inj', 'away_usage_percentage_inj','away_offensive_win_shares_inj','away_defensive_win_shares_inj','away_value_over_replacement_inj']        
    #home_features = ['home_pace','home_assist_percentage','home_assists','home_block_percentage','home_blocks','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_effective_field_goal_percentage','home_field_goal_attempts','home_field_goal_percentage','home_field_goals','home_free_throw_attempts','home_free_throw_percentage','home_free_throws','home_offensive_rating','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_three_point_field_goals','home_total_rebound_percentage','home_total_rebounds','home_true_shooting_percentage','home_turnover_percentage','home_turnovers','home_two_point_field_goal_attempts','home_two_point_field_goal_percentage','home_two_point_field_goals','home_minutes_played_inj','home_usage_percentage_inj','home_offensive_win_shares_inj','home_defensive_win_shares_inj','home_value_over_replacement_inj']
    
    away_features = ['away_pace','away_assist_percentage','away_block_percentage','away_defensive_rating','away_defensive_rebound_percentage','away_field_goal_percentage','away_free_throw_attempts','away_free_throw_percentage','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_total_rebound_percentage','away_turnover_percentage','away_turnovers','away_minutes_played_inj', 'away_usage_percentage_inj','away_offensive_win_shares_inj', 'away_defensive_win_shares_inj','away_value_over_replacement_inj']        
    home_features = ['home_pace','home_assist_percentage','home_block_percentage','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_field_goal_percentage','home_free_throw_attempts','home_free_throw_percentage','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_total_rebound_percentage','home_turnover_percentage','home_turnovers','home_minutes_played_inj','home_usage_percentage_inj','home_offensive_win_shares_inj', 'home_defensive_win_shares_inj','home_value_over_replacement_inj']
    
    
    class_features = away_features + home_features
    output_label = ['Winner']
    
    use_PCA = True
    percentage_variance_explained = 0.9
    
    
#class_params = [300, 0.01, 5] or [80, 3]
model = 'opt'
if model =='opt':
    #For optimized model
    class_params = [80, 3]
elif model =='original':
    #For original model
    class_params = [300, 5]
classifier_model = 'RF'

#Select the regression features & model hyperparams
reg_features = ['V ML', 'H ML', 'Pred Probs V', 'Pred Probs H']
reg_label = ['Classifier Profit']

#300, 5 or 100, 5. Seems to be fairly insensitive to these
reg_params = [100, 5]   
reg_threshold = 0.1

#Train and test the layered model
class_model, profit_reg_model, testing_gains_reg, pca = pf.layered_model_TrainTest(training_df, testing_df, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scraped, use_PCA, percentage_variance_explained)

if hist:
    bins = np.arange(-20,200, 5)
    plt.figure()
    plt.hist(testing_gains_reg, bins = bins)
    plt.title('Gains Histogram, Testing')
    """
    sum_nocap_test = sum(testing_gains_reg)
    for i in range(len(testing_gains_reg)):
        if testing_gains_reg[i]>25:
            testing_gains_reg[i] = 25
    sum_capped_test = sum(testing_gains_reg)
    print('With cap at 25:', sum_capped_test, ', without cap:', sum_nocap_test)
    """

wins = 0
win_tally = 0
loss_tally = 0
for i in range(len(testing_gains_reg)):
    if testing_gains_reg[i] > 0:
        wins += 1
        win_tally += testing_gains_reg[i]
    else:
        loss_tally += -testing_gains_reg[i]
        
avg_win = win_tally/len(testing_gains_reg)
avg_loss = loss_tally/len(testing_gains_reg)
R = avg_win/avg_loss
winpct = wins/len(testing_gains_reg)
print('win%:', winpct)
print('R:', R)

#Validate the model using the unseen 2017 data
validation_test = True
if validation_test:
    fixed_wager = False
    val_gains_reg = pf.layered_model_validate(validation_df, class_features, output_label, class_model, reg_features, profit_reg_model, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scraped, use_PCA, pca)
    
    wins = 0
    win_tally = 0
    loss_tally = 0
    for i in range(len(val_gains_reg)):
        if val_gains_reg[i] > 0:
            wins += 1
            win_tally += val_gains_reg[i]
        else:
            loss_tally += -val_gains_reg[i]
            
    avg_win = win_tally/len(val_gains_reg)
    avg_loss = loss_tally/len(val_gains_reg)
    R = avg_win/avg_loss
    winpct = wins/len(val_gains_reg)
    print('win%:', winpct)
    print('R:', R)
    
    if hist:
        plt.figure()
        plt.hist(val_gains_reg, bins = bins)
        plt.title('Gains Histogram, Validation')
        """
        sum_nocap_val = sum(val_gains_reg)
        for i in range(len(val_gains_reg)):
            if val_gains_reg[i]>25:
                val_gains_reg[i] = 25
        sum_capped_val = sum(val_gains_reg)
        print('With cap at 25:', sum_capped_val, ', without cap:', sum_nocap_val)
        """
#Given some new games, determine what bets to place
new_bets = False
if new_bets:
    account = 100
    fixed_wager = False
    reg_threshold = 0.1
    todays_index = 136 #index of first game today
    #Analyze the upcoming games and decide when/how much to bet
    current_data_df = data2020_df[todays_index:]
    bet_placed_indices, recommended_wagers, recommended_bets = pf.make_new_bets(current_data_df, class_features, output_label, class_model, reg_features, reg_label, profit_reg_model, reg_threshold, fixed_wager, wager_pct, wager_crit, account, use_PCA, pca)
    
    #load the bet tracking csv, append the new bets, and save back to csv
    bet_tracking_df = current_data_df.copy()        
    bet_tracking_df['Recommended Wager'] = recommended_wagers
    bet_tracking_df['Recommended Winner'] = recommended_bets

    update_bet_tracking = True
    if update_bet_tracking:
        if model =='opt':
            bet_tracking_df_old = pd.read_csv('Data/bet_tracking2021_optModel.csv')
        elif model=='original':
            bet_tracking_df_old = pd.read_csv('Data/bet_tracking2021_originalModel.csv')
        
        bet_tracking_df = pd.concat([bet_tracking_df_old, bet_tracking_df])
        
    #bet_tracking_df.to_csv('Data/bet_tracking2021_optModel.csv', index=False)
    #bet_tracking_df.to_csv('Data/bet_tracking2021_originalModel.csv', index=False)
    
    
