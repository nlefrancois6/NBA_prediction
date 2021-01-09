#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:15:30 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def get_next_day(day, month, year):
        
    months30 = [4,6,9,11]
    months31 = [1,3,5,7,8,10,12]
    leapyears = [2012, 2016, 2020]
    if month in months30:
        if day<30:
            day += 1
        else:
            month += 1
            day = 1
    elif month in months31:
        if day<31:
            day += 1
        else:
            if month == 12:
                year += 1
                month = 1
                day = 1
            else:
                month += 1
                day = 1           
    else:
        if year in leapyears:
            if day<29:
                day += 1
            else:
                month += 1
                day = 1
        else:
            if day<28:
                day += 1
            else:
                month += 1
                day = 1
    return day, month, year

def expected_gain(pred_prob, odds_v, odds_h):
    """
    Calculate the expected value of a bet
    
    pred_prob: Array of form [v_win_prob, v_lose_prob] with the model prediction probs
    
    odds_v: the moneyline odds on visitors
    
    odds_h: the moneyline odds on home
    """
    wager = 10
    
    #Get the predicted winner
    if pred_prob[0] > pred_prob[1]:
        visitor_win_pred = 'Win'
        correct_prob = pred_prob[0]
        wrong_prob = pred_prob[1]
    else:
        visitor_win_pred = 'Loss'
        correct_prob = pred_prob[1]
        wrong_prob = pred_prob[0]
    #Find the gain we would get if our prediction is correct
    if visitor_win_pred == 'Win':
        #odds[0] is odds on visitor win
        if odds_v > 0:
            gain_correct_pred = odds_v*(wager/100)
        else:
            gain_correct_pred = 100*(wager/(-odds_v))
    if pred_prob[0] < pred_prob[1]:
        #odds[1] is odds on home win
        if odds_h > 0:
            gain_correct_pred = odds_h*(wager/100)
        else:
            gain_correct_pred = 100*(wager/(-odds_h))
    #If our prediction is wrong, we lose the wager
    gain_wrong_pred = -wager
    #The expected gain is equal to each of the two possible gain outcomes multiplied
    #by the probability of the outcome, as determined by the model confidences
    exp_gain = gain_correct_pred*correct_prob + gain_wrong_pred*wrong_prob
    
    return exp_gain

def calc_Profit(account, wager_pct, fixed_wager, wager_crit, winner_prediction, winner_actual, moneyline_odds, regression_threshold, reg_profit_exp, expectation_threshold=False, scrape=False):
    """
    account: total money in the account at the start
    
    wager_pct: the amount wagered on each game as a fraction of the account. 
        float [0,1]
        
    fixed_wager: if True, just bet 10$ on every game. If False, use wager_pct to 
        calculate the amount to bet. In order to get meaningful/normalized labelling for
        the regression, need to set this to False.
    
    winner_prediction: the prediction of whether visiting team will win or lose.
        Possible values are 'Win' and 'Loss'
    
    winner_actual: the actual result of whether visiting team won or lost.
        Possible values are 'Win' and 'Loss' (might need to handle 'push')
    
    moneyline_odds: the moneyline odds given for visiting & home teams
        Not sure of format yet but probably (numGames,2) array with [V odds, H odds]
        Might need to apply a conversion for negative (ie favourite) odds, or handle the negative here
    
    pred_probs: the models confidence in each outcome. [prob V win, prob H win]
    
    expectation_threshold: If False, bet on every game. If a number, only bet when the 
        expected value of the bet is above that number.
    
    regression_threshold: If False, bet on every game. If a number, only bet when the 
        regression's expected value of the bet is above that number.
    
    reg_profit_exp: the regression's expected value of each bet
    
        
    Returns account_runningTotal, an array containing the total money we have after each game,
    and gains_store, a list of the gains from each bet.
    
    """
    #wins = 1
    #losses = 1
    first = True
    account_runningTotal = [account]
    gains_store = []
    gain = 0
    numGames = len(winner_prediction)
    for i in range(numGames):
        #By default, bet on the game (ie threshold_met = True)
        threshold_met = True
        #Check if an expected gain threshold is set
        if expectation_threshold != False:
            #Calculate the expected gain and check if it is above the set threshold
            #exp_gain = expected_gain(pred_probs[i], moneyline_odds[i,0], moneyline_odds[i,1])
            print('Expectation threshold is obsolete. Use regression threshold instead.')
            #if exp_gain < expectation_threshold:
                #If the threshold is not met, do not bet on the game
                #threshold_met = False
        #Check if a regression_threshold is set
        if regression_threshold != False:
            #Get the expected profit from the regression model and check if it's above the threshold
            exp_gain = reg_profit_exp[i]
            if exp_gain < regression_threshold:
                #If the threshold is not met, do not bet on the game
                threshold_met = False
        else:
            exp_gain = 5
        if threshold_met == True:
            if fixed_wager == False:
                #Seem to have best results with sqrt scaling
                #wager = wager_pct*100*np.exp(-exp_gain)
                if wager_crit == 'log':
                    wager = wager_pct*100*np.log(exp_gain/4+1.0001)
                if wager_crit == 'kelly':
                    #Could use these numbers as an initial value and continue to track & update them
                    #win_pct = wins/(wins+losses)
                    #w_l_ratio = avg_win/avg_loss
                    win_pct = 0.56
                    w_l_ratio = 1.28
                    k_pct = win_pct - (1-win_pct)/w_l_ratio #0.22
                    if first==True:
                        print(k_pct)
                        first = False
                    wager = k_pct*100 #could further multiply this by some f(exp_gain)
                    
                if wager_crit == 'sqrt':
                    wager = wager_pct*100*np.sqrt(exp_gain/5)
                    
                if wager > 20:
                    wager = 20
            else:
                wager = 10
            #If our prediction was correct, calculate the winnings
            if winner_actual[i] == winner_prediction[i]:
                #wins += 1
                if scrape:
                    if winner_prediction[i] == 'V':
                        #odds[0] is odds on visitor win
                        if moneyline_odds[i,0]>0:
                            gain = moneyline_odds[i,0]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,0]))
                    if winner_prediction[i] == 'H':
                        #odds[1] is odds on home win
                        if moneyline_odds[i,1]>0:
                            gain = moneyline_odds[i,1]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,1]))
                else:
                    if winner_prediction[i] == 'Win':
                        #odds[0] is odds on visitor win
                        if moneyline_odds[i,0]>0:
                            gain = moneyline_odds[i,0]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,0]))
                    if winner_prediction[i] == 'Loss':
                        #odds[1] is odds on home win
                        if moneyline_odds[i,1]>0:
                            gain = moneyline_odds[i,1]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,1]))
            #If our prediction was wrong, lose the wager
            else:
                #losses += 1
                gain = -wager
        
            account = account + gain
            account_runningTotal.append(account)
            gains_store.append(gain)
        
    return account_runningTotal, gains_store

def evaluate_model_profit(preds, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, fixed_wager, wager_crit, plot_gains, dummy_odds = False, regression_threshold=False, reg_profit_exp = False, scrape = False):
    """
    Take the predictions made by a model and a testing set with labels & the odds, calculate the
    final account balance and plot the account balance. Also return gains_store, a list of
    the gains from each bet.
    """
    
    account = 0
    winner_prediction = preds
    if scrape:
        winner_actual = testing_label['Winner'].array
    else:
        winner_actual = testing_label['teamRslt'].array
    
    #Get the ML odds for each game, either from the data or using dummy odds
    moneyline_odds = np.zeros([len(winner_prediction),2])
    if dummy_odds:
        for i in range(len(winner_prediction)):
            moneyline_odds[i,0] = 170
            moneyline_odds[i,1] = -200
    else:
        for i in range(len(winner_prediction)):
            moneyline_odds[i,0] = testing_odds_v.iloc[i]
            moneyline_odds[i,1] = testing_odds_h.iloc[i]

    account_runningTotal, gains_store = calc_Profit(account, wager_pct, fixed_wager, wager_crit, winner_prediction, winner_actual, moneyline_odds, regression_threshold, reg_profit_exp, expectation_threshold = min_exp_gain, scrape = scrape)
    
    #print('Final Account Balance after ', len(account_runningTotal), ' games: ', account_runningTotal[-1])
    
    if plot_gains:
        print('Final Account Balance after ', len(account_runningTotal), ' games: ', account_runningTotal[-1])
    
        plt.figure()
        plt.plot(account_runningTotal)
        plt.hlines(account, 0, len(account_runningTotal),linestyles='dashed')
        plt.xlabel('Games')
        plt.ylabel('Account Total')
        plt.title('Betting Performance of Our Model')
    
    return account_runningTotal, gains_store

def model_feature_importances(plot_features, model, features):
    """
    Plot and return the feature importances vector of a model
    """
    feature_importance = model.feature_importances_.tolist()
    if plot_features:
        plt.figure()
        plt.bar(features,feature_importance)
        plt.title('Feature Importance')
        plt.xticks(rotation='vertical')
        plt.show()
    
    return feature_importance

def avg_previous_num_games(df, num_games, window, home_features, away_features, team_list, scrape=False):
    # This function changes each stat to be the average of the last num_games for each team, and shifts it one so it does not include the current stats and drops the first num_games that become null
    if scrape:
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    else: 
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    return df.dropna()
"""
def avg_previous_num_games_scrape(df, num_games, window, home_features, away_features, team_list):
    # This function changes each stat to be the average of the last num_games for each team, and shifts it one so it does not include the current stats and drops the first num_games that become null
    for col in home_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            if window == 'flat':
                df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
            if window == 'gauss':
                df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    for col in away_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            if window == 'flat':
                df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
            if window == 'gauss':
                df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    return df.dropna()
"""
def avg_season(df, home_features, away_features, team_list, scrape=False):
    # This function changes each stat to be the season average for each team going into the game, and shifts it one so it does not include the current stats and drops the first num_games that become null
    if scrape:
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).expanding(min_periods=15).mean()
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).expanding(min_periods=15).mean()   
    else:
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).expanding(min_periods=15).mean()
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).expanding(min_periods=15).mean()
    return df.dropna()
"""
def avg_season_scrape(df, home_features, away_features, team_list):
    # This function changes each stat to be the season average for each team going into the game, and shifts it one so it does not include the current stats and drops the first num_games that become null
    for col in home_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).expanding(min_periods=15).mean()
    for col in away_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).expanding(min_periods=15).mean()
    return df.dropna()
"""
#Same-day games are not necessarily aligned between the two data sets, so I need to get
#them manually from odds_df and then store them as two new columns in stats_df
def get_sameday_games(stats_df_day, season, odds_df):
    """
    Take date in format 2015-10-27 and convert to format 1027 to access odds from
    all games on that day in the given season
    """

    odds_format_day = int(stats_df_day[5:7] + stats_df_day[8:10])
    sameyear_odds_df = odds_df.loc[odds_df['Year'] == season]
    sameday_odds_df = sameyear_odds_df.loc[sameyear_odds_df['Date'] == odds_format_day]
    
    return sameday_odds_df

def get_matching_game_odds(stats_df_day, season, v_team, h_team, odds_df):
    """
    Take the date and the two teams for a game in stats_df, and get the rows of odds_df 
    containing the two teams from that game. Extract the ML odds from each row/team
    """
    sameday_odds_df = get_sameday_games(stats_df_day, season, odds_df)
    
    #Need to make sure 'Team' column is in same format
    v_odds_row = sameday_odds_df.loc[sameday_odds_df['Team'] == v_team]
    h_odds_row = sameday_odds_df.loc[sameday_odds_df['Team'] == h_team]
    
    v_MLodds = v_odds_row['ML']
    h_MLodds = h_odds_row['ML']
    
    return v_MLodds, h_MLodds

def get_matching_game_score(stats_df_day, season, v_team, h_team, odds_df):
    """
    Take the date and the two teams for a game in stats_df, and get the rows of odds_df 
    containing the two teams from that game. Extract the score of each team from each row/team
    """
    sameday_odds_df = get_sameday_games(stats_df_day, season, odds_df)
    
    #Need to make sure 'Team' column is in same format
    v_odds_row = sameday_odds_df.loc[sameday_odds_df['Team'] == v_team]
    h_odds_row = sameday_odds_df.loc[sameday_odds_df['Team'] == h_team]
    
    v_score = v_odds_row['Final']
    h_score = h_odds_row['Final']
    
    return v_score, h_score


def get_season_odds_matched(stats_df_year, odds_df):
    """
    Given stats_df for a year, find the corresponding odds for every game and return
    the v and h odds in two arrays ready to be added to the df. Also return the number
    of games that encountered the size==0 error.
    """
    v_odds_list = []
    h_odds_list = []
    v_score_list = []
    h_score_list = []
    broke_count = 0
    for i in range(len(stats_df_year)):
        season = stats_df_year['Season'].iloc[i]
        game_date = stats_df_year['gmDate'].iloc[i]
        v_team = stats_df_year['teamAbbr'].iloc[i]
        h_team = stats_df_year['opptAbbr'].iloc[i]
        v_odds, h_odds = get_matching_game_odds(game_date, season, v_team, h_team, odds_df)
        v_score, h_score = get_matching_game_score(game_date, season, v_team, h_team, odds_df)
    
        if (v_odds.size == 0) or (h_odds.size == 0):
            broke_count = broke_count + 1
            #Sometimes the odds have size 0. I think this is probably because the game isn't found
            #Maybe I'll want to drop that one, but for now just setting it to a neutral odds
            #should be fine
            v_odds_num = 100
            h_odds_num = 100
        else:
            v_odds_num = v_odds.iloc[0]
            h_odds_num = h_odds.iloc[0]
            
        v_score_num = v_score.iloc[0]
        h_score_num = h_score.iloc[0]
        
        v_odds_list.append(v_odds_num)
        h_odds_list.append(h_odds_num)
        v_score_list.append(v_score_num)
        h_score_list.append(h_score_num)
    
    return v_odds_list, h_odds_list, v_score_list, h_score_list, broke_count

def get_season_odds_matched_scrape(stats_df_year, odds_df):
    """
    Given stats_df for a year, find the corresponding odds for every game and return
    the v and h odds in two arrays ready to be added to the df. Also return the number
    of games that encountered the size==0 error.
    """
    v_odds_list = []
    h_odds_list = []
    v_score_list = []
    h_score_list = []
    broke_count = 0
    for i in range(len(stats_df_year)):
        noOdds = False
        season = stats_df_year['Season'].iloc[i]
        game_date = stats_df_year['gmDate'].iloc[i]
        v_team = stats_df_year['away_abbr'].iloc[i]
        h_team = stats_df_year['home_abbr'].iloc[i]
        v_odds, h_odds = get_matching_game_odds(game_date, season, v_team, h_team, odds_df)
        v_score, h_score = get_matching_game_score(game_date, season, v_team, h_team, odds_df)
        
        if (v_odds.size == 0) or (h_odds.size == 0):
            broke_count = broke_count + 1
            #Sometimes the odds have size 0. I think this is probably because the game isn't found
            #Maybe I'll want to drop that one, but for now just setting it to a neutral odds
            #should be fine
            v_odds_num = 0
            h_odds_num = 0
        else:
            v_odds_num = v_odds.iloc[0]
            h_odds_num = h_odds.iloc[0]
        
        if (v_score.size == 0) or (h_score.size == 0):
            noOdds = True
        
        if noOdds == False:
            v_score_num = v_score.iloc[0]
            h_score_num = h_score.iloc[0]
        else:
            v_score_num = 0
            h_score_num = 0
        
        v_odds_list.append(v_odds_num)
        h_odds_list.append(h_odds_num)
        v_score_list.append(v_score_num)
        h_score_list.append(h_score_num)
    
    return v_odds_list, h_odds_list, v_score_list, h_score_list, broke_count


def gbcModel(training_features, training_label, testing_label, testing_features, n_est, learn_r, max_d):
    #Normalize the data
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)
    scaler.transform(testing_features)
    
    #Train a Gradient Boosting Machine on the data
    gbc = ensemble.GradientBoostingClassifier(n_estimators = n_est, learning_rate = learn_r, max_depth=max_d, subsample=1.0)
    gbc.fit(training_features, training_label)

    #Predict the outcome from our test set and evaluate the prediction accuracy for each model
    predGB = gbc.predict(testing_features) 
    pred_probsGB = gbc.predict_proba(testing_features) #probability of [results==True, results==False]
    accuracyGB = metrics.accuracy_score(testing_label, predGB)
    
    return gbc, predGB, pred_probsGB, accuracyGB

def rfcModel(training_features, training_label, testing_label, testing_features, n_est, rs, max_d):
    #Normalize the data
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)
    scaler.transform(testing_features)
    
    #Train a Random Forest on the data
    rfc = ensemble.RandomForestClassifier(n_estimators = n_est, max_depth=max_d, random_state = rs)
    rfc.fit(training_features, training_label)

    #Predict the outcome from our test set and evaluate the prediction accuracy for each model
    predRF = rfc.predict(testing_features) 
    pred_probsRF = rfc.predict_proba(testing_features) #probability of [results==True, results==False]
    accuracyRF = metrics.accuracy_score(testing_label, predRF)
    
    return rfc, predRF, pred_probsRF, accuracyRF

def kerasModel(training_features, training_label, testing_label, testing_features):
    encoder = LabelEncoder()
    encoder.fit(training_label)
    training_label_enc = encoder.transform(training_label)
    testing_label_enc = encoder.transform(testing_label)
    
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)
    scaler.transform(testing_features)
    
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(60, activation='relu'),
      tf.keras.layers.Dense(30, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(training_features.values, training_label_enc, epochs=10)
    accuracy = model.evaluate(testing_features, testing_label_enc)
    
    pred_probs = model.predict(testing_features)
    preds = []
    for i in range(len(pred_probs)):
        if pred_probs[i]>0.5:
            #not sure if it will always get encoded the same way. Should be a way to get the map from the encoder directly
            preds.append(['V'])
        else:
            preds.append(['H'])
    
    
    return model, preds, pred_probs, accuracy

def get_score_prediction(data_features, v_score_model, h_score_model):
    """
    Take two score prediction models and get their predictions on the games
    in data_features
    
    Outputs:    
        pred_winner: if 1, V win predicted. if 0, V loss predicted
        
        pred_margin: if positive, predicted margin of victory for V. if negative, 
            predicted margin of loss for V.
            
        v_score_pred, h_score_pred: predicted score for each team
            
    """
    v_score_pred = v_score_model.predict(data_features)
    h_score_pred = h_score_model.predict(data_features)
    #Get predicted winner and predicted margin on testing set
    numGames = len(v_score_pred)
    pred_winner = []
    for i in range(numGames):
        if v_score_pred[i] > h_score_pred[i]:
            pred_winner.append('Win')
        else:
            pred_winner.append('Loss')
    
    pred_margin = v_score_pred - h_score_pred
    
    return pred_winner, pred_margin, v_score_pred, h_score_pred

def score_prediction_model(training_features, training_label_v, training_label_h, testing_features, n_estimators, max_depth):
    """
    Fit two regression models to predict v points and h points, which can be used to
    predict spread, O/U, or ML
    """
    random_state = 1
    #Fit a regression model to predict visitor score
    v_score_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    v_score_model.fit(training_features, training_label_v)
    #Fit a regression model to predict home score
    h_score_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    h_score_model.fit(training_features, training_label_h)
    #Get predictions for score on testing set
    pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test = get_score_prediction(testing_features, v_score_model, h_score_model)
        
    return pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test, v_score_model, h_score_model

def layered_model_doubleReg_TrainTest(training_df, testing_df, class_features, output_label, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct):
    """
    Train and test the classification-regression layered model
    
    Inputs:
        training_df: Dataframe containing the training data
        
        testing_df: Dataframe containing the testing data
        
        class_features: array, features to use for the classifier model
        
        output_label: string, target output of the classifier model. Should be 'teamRslt'
        
        classifier_model: string, which classifier model to use
        
        class_params: array, hyperparameters for the classifier model
        
        reg_features: array, features to use for the regression model
        
        reg_label: string, target output of the regression model. Should be 'Classifier Profit'
        
        reg_params: array, hyperparameters for the regression model
        
        reg_threshold: float, minimum regression expectation required to place a bet
        
        plot_gains: boolean, if True plot the gains on the testing and validation set
            and print the final account balance w/ # of bets placed
            
        fixed_wager: boolean, if True just bet 10$ on every game
        
        wager_pct: float, if fixed_wager==False, bet balance*wager_pct
        
    Outputs:
        class_model: trained classification model object
            
        classifier_feature_importance: array, feature importance weightings for class_model
            
        profit_reg_model: trained regression model object
            
        testing_gains_reg: array, gain/loss from each game we bet upon
    """
    
    #Define the features (input) and label (prediction output) for training set
    training_features = training_df[class_features]
    #training_label = training_df[output_label]
    #training_odds_v = training_df['V ML'] #might not be needed
    #training_odds_h = training_df['H ML'] #might not be needed

    #Define features and label for testing set
    testing_features = testing_df[class_features]
    testing_label = testing_df[output_label]
    testing_odds_v = testing_df['V ML']
    testing_odds_h = testing_df['H ML']
    
    """
    This gives different outputs than the classifier, and will go into the profit regression
    differently. I think I should just write a whole new function to do this triple regression
    2-layer model, but that'll require me to change the name of this function and I don't
    want to mess with that right now.
    """
    #Train two regression models to predict V score and H score, use this to get
    #the predicted winner and predicted margin of victory
    training_label_v = training_df['V Score']
    training_label_h = training_df['H Score']
        
    n_estimators = 300
    max_depth = 5
        
    pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test, v_score_model, h_score_model = score_prediction_model(training_features, training_label_v, training_label_h, testing_features, n_estimators, max_depth)
     
    
    min_exp_gain = False
    plot_gains_score = False #This is hard-coded since I don't think I need it anymore
    running_account, testing_gains = evaluate_model_profit(pred_winner_test, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, fixed_wager, plot_gains_score, dummy_odds = False)
    if plot_gains_score:
        plt.title('Score Regression Model Profit, Testing')


    d = {'Classifier Profit': testing_gains, 'Pred Score V': v_score_pred_test, 'Pred Score H': h_score_pred_test, 'Prediction': pred_winner_test}
    reg_df = pd.DataFrame(data=d)

    reg_df['V ML'] = testing_odds_v.reset_index().drop(columns = ['index'])
    reg_df['H ML'] = testing_odds_h.reset_index().drop(columns = ['index'])
    
    #Separate the 2nd 1/2 of the data into training and testing data (I might end up using another year of data for testing instead)
    training_reg_df = reg_df.sample(frac=0.5, random_state=1)
    indlist_reg=list(training_reg_df.index.values)

    testing_reg_df = reg_df.copy().drop(index=indlist_reg)

    #Define the features and label for training set
    training_reg_features = training_reg_df[reg_features]
    training_reg_label = training_reg_df[reg_label]

    #Define features and label for testing set
    testing_reg_features = testing_reg_df[reg_features]
    #testing_reg_label = testing_reg_df[reg_label] #might not be needed
    
    #Create and train a regression model to predict the profit based on odds and classifier confidence
    #n_estimators = 100
    #max_depth = 3
    n_estimators = reg_params[0]
    max_depth = reg_params[1]
    
    random_state = 1
    
    profit_reg_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    #Train the model on the third 1/4 of the data
    profit_reg_model.fit(training_reg_features, training_reg_label)
    #Get the expected profit on the remaining 1/4 of the data
    expected_profit_testing = profit_reg_model.predict(testing_reg_features)

    #Do some dumb data formatting to get things in arrays even though alot of this doesn't get
    #used when we use the regression threshold instead of expected value
    preds_reg_testing = testing_reg_df['Prediction'].reset_index().drop(columns = ['index'])
    
    preds_reg_testing_arr = []
    for i in range(len(preds_reg_testing)):
        preds_reg_testing_arr.append(preds_reg_testing['Prediction'][i])
        
    testing_reg_odds_v = testing_reg_df['V ML'].reset_index().drop(columns = ['index'])
    testing_reg_odds_h = testing_reg_df['H ML'].reset_index().drop(columns = ['index'])

    #Evaluate the layered model profit on the remaining 1/4 of the testing data
    
    #Calculate the profit when we only bet on games the regression expectation favours 
    running_account_reg, testing_gains_reg = evaluate_model_profit(preds_reg_testing_arr, testing_label, testing_reg_odds_v, testing_reg_odds_h, min_exp_gain, wager_pct, fixed_wager, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_testing)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Testing Data')
    
    return v_score_model, h_score_model, profit_reg_model, testing_gains_reg


def layered_model_TrainTest(training_df, testing_df, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scrape):
    """
    Train and test the classification-regression layered model
    
    Inputs:
        training_df: Dataframe containing the training data
        
        testing_df: Dataframe containing the testing data
        
        class_features: array, features to use for the classifier model
        
        output_label: string, target output of the classifier model. Should be 'teamRslt'
        
        classifier_model: string, which classifier model to use
        
        class_params: array, hyperparameters for the classifier model
        
        reg_features: array, features to use for the regression model
        
        reg_label: string, target output of the regression model. Should be 'Classifier Profit'
        
        reg_params: array, hyperparameters for the regression model
        
        reg_threshold: float, minimum regression expectation required to place a bet
        
        plot_gains: boolean, if True plot the gains on the testing and validation set
            and print the final account balance w/ # of bets placed
            
        fixed_wager: boolean, if True just bet 10$ on every game
        
        wager_pct: float, if fixed_wager==False, bet balance*wager_pct
        
    Outputs:
        class_model: trained classification model object
            
        classifier_feature_importance: array, feature importance weightings for class_model
            
        profit_reg_model: trained regression model object
            
        testing_gains_reg: array, gain/loss from each game we bet upon
    """
    
    #Define the features (input) and label (prediction output) for training set
    training_features = training_df[class_features]
    training_label = training_df[output_label]
    #training_odds_v = training_df['V ML'] #might not be needed
    #training_odds_h = training_df['H ML'] #might not be needed

    #Define features and label for testing set
    testing_features = testing_df[class_features]
    testing_label = testing_df[output_label]
    testing_odds_v = testing_df['V ML']
    testing_odds_h = testing_df['H ML']
    
    if classifier_model == 'GB':
        #Train a Gradient Boosting Machine on the data, predict the outcomes, and evaluate accuracy
        #100, 0.02, 5 seems to work well. Also 500, 0.02, 5.
        #n_estimators = 500
        #learning_rate = 0.02
        #max_depth = 5
        n_estimators = class_params[0]
        learning_rate = class_params[1]
        max_depth = class_params[2]
        
        class_model, pred_class_test, pred_probs_class_test, accuracy_class_test = gbcModel(training_features, training_label, testing_label, testing_features, n_estimators, learning_rate, max_depth)
    elif classifier_model == 'RF':
        #Train a Random Forest Classifier on the data, predict the outcomes, and evaluate accuracy
        #n_estimators = 100
        #max_depth = 1
        n_estimators = class_params[0]
        max_depth = class_params[1]
        
        random_state = 1
        class_model, pred_class_test, pred_probs_class_test, accuracy_class_test = rfcModel(training_features, training_label, testing_label, testing_features, n_estimators, random_state, max_depth)
    elif classifier_model == 'Keras':
        #Probably need to handle the label encoding/decoding in other places
        class_model, pred_class_test, pred_probs_class_test, accuracy_class_test = kerasModel(training_features, training_label, testing_label, testing_features)
    """
    This gives different outputs than the classifier, and will go into the profit regression
    differently. I think I should just write a whole new function to do this triple regression
    2-layer model, but that'll require me to change the name of this function and I don't
    want to mess with that right now.
    elif classifier_model == 'Score Predict':
        
        training_label_v = training_df['V Score']
        training_label_h = training_df['H Score']
        
        n_estimators = 300
        max_depth = 5
        
        pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test, v_score_model, h_score_model = score_prediction_model(training_features, training_label_v, training_label_h, testing_features, n_estimators, max_depth)
     """   
        
    #Feature importance plots
    plot_features = False
    #classifier_feature_importance = model_feature_importances(plot_features, class_model, class_features)

    min_exp_gain = False
    
    plot_gains_class = False #This is hard-coded since I don't think I need it anymore
    fixed_wager_class = True
    running_account, testing_gains = evaluate_model_profit(pred_class_test, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, fixed_wager_class, wager_crit, plot_gains_class, dummy_odds = False, scrape = scrape)
    if plot_gains_class:
        plt.title('Classifier Model Profit, Testing')

    if classifier_model == 'Keras':
        pred_probs_h = 1 - pred_probs_class_test
        d = {'Classifier Profit': testing_gains, 'Pred Probs V': pred_probs_class_test[:,0], 'Pred Probs H': pred_probs_h[:,0], 'Prediction': pred_class_test}
    else:
        #print(pred_probs_class_test.shape)
        d = {'Classifier Profit': testing_gains, 'Pred Probs V': pred_probs_class_test[:,0], 'Pred Probs H': pred_probs_class_test[:,1], 'Prediction': pred_class_test}
    reg_df = pd.DataFrame(data=d)
    #print(reg_df.shape)

    reg_df['V ML'] = testing_odds_v.reset_index().drop(columns = ['index'])
    reg_df['H ML'] = testing_odds_h.reset_index().drop(columns = ['index'])
    
    #Separate the 2nd 1/2 of the data into training and testing data (I might end up using another year of data for testing instead)
    training_reg_df = reg_df.sample(frac=0.5, random_state=1)
    indlist_reg=list(training_reg_df.index.values)

    testing_reg_df = reg_df.copy().drop(index=indlist_reg)

    #Define the features and label for training set
    training_reg_features = training_reg_df[reg_features]
    training_reg_label = training_reg_df[reg_label]

    #Define features and label for testing set
    testing_reg_features = testing_reg_df[reg_features]
    #testing_reg_label = testing_reg_df[reg_label] #might not be needed
    
    #Create and train a regression model to predict the profit based on odds and classifier confidence
    #n_estimators = 100
    #max_depth = 3
    n_estimators = reg_params[0]
    max_depth = reg_params[1]
    
    random_state = 1
    
    profit_reg_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    #Train the model on the third 1/4 of the data
    profit_reg_model.fit(training_reg_features, training_reg_label)
    #Get the expected profit on the remaining 1/4 of the data
    expected_profit_testing = profit_reg_model.predict(testing_reg_features)

    #Do some dumb data formatting to get things in arrays even though alot of this doesn't get
    #used when we use the regression threshold instead of expected value
    preds_reg_testing = testing_reg_df['Prediction'].reset_index().drop(columns = ['index'])
    pred_probs_reg_testing_v = testing_reg_df['Pred Probs V'].reset_index().drop(columns = ['index'])
    pred_probs_reg_testing_h = testing_reg_df['Pred Probs H'].reset_index().drop(columns = ['index'])

    preds_reg_testing_arr = []
    pred_probs_reg_testing = []
    for i in range(len(preds_reg_testing)):
        preds_reg_testing_arr.append(preds_reg_testing['Prediction'][i])
        pred_probs_reg_testing.append([pred_probs_reg_testing_v['Pred Probs V'][i], pred_probs_reg_testing_h['Pred Probs H'][i]])

    testing_reg_odds_v = testing_reg_df['V ML'].reset_index().drop(columns = ['index'])
    testing_reg_odds_h = testing_reg_df['H ML'].reset_index().drop(columns = ['index'])

    #Evaluate the layered model profit on the remaining 1/4 of the testing data
    
    #Calculate the profit when we only bet on games the regression expectation favours 
    running_account_reg, testing_gains_reg = evaluate_model_profit(preds_reg_testing_arr, testing_label, testing_reg_odds_v, testing_reg_odds_h, min_exp_gain, wager_pct, fixed_wager, wager_crit, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_testing, scrape = scrape)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Testing Data')
    
    return class_model, profit_reg_model, testing_gains_reg



def layered_model_validate(validation_data_df, class_features, output_label, class_model, reg_features, profit_reg_model, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scrape):
    """
    Validate the layered model using an unseen dataset. Inputs are mostly the same as for 
    layered_model_TestTrain, with class_model and profit_reg_model the two trained model
    objects that make up our layered model.
    """
    
    #Format the validation data
    validation_class_features = validation_data_df[class_features]
    validation_label = validation_data_df[output_label]
    validation_odds_v = validation_data_df['V ML']
    validation_odds_h = validation_data_df['H ML']
    
    #Get classifier predictions on validation data & evaluate the gains
    pred_class_val = class_model.predict(validation_class_features) 
    pred_probs_class_val = class_model.predict_proba(validation_class_features)
    
    min_exp_gain = False
    plot_gains_class = False
    fixed_wager_class = True
    running_account_val, gains_val_class = evaluate_model_profit(pred_class_val, validation_label, validation_odds_v, validation_odds_h, min_exp_gain, wager_pct, fixed_wager_class, wager_crit, plot_gains_class, dummy_odds = False, scrape = scrape)
    if plot_gains_class:
        plt.title('Classifier Model Profit, Validation Data')
    
    #Data formatting for the regression
    d_val = {'Classifier Profit': gains_val_class, 'Pred Probs V': pred_probs_class_val[:,0], 'Pred Probs H': pred_probs_class_val[:,1], 'Prediction': pred_class_val}
    reg_val_df = pd.DataFrame(data=d_val)

    reg_val_df['V ML'] = validation_odds_v.reset_index().drop(columns = ['index'])
    reg_val_df['H ML'] = validation_odds_h.reset_index().drop(columns = ['index'])
    
    reg_val_features = reg_val_df[reg_features]
    
    #Get regression predictions on validation data
    expected_profit_val = profit_reg_model.predict(reg_val_features)
    
    #Data formatting for layered model profit evaluation
    preds_val_reg = reg_val_df['Prediction'].reset_index().drop(columns = ['index'])
    pred_probs_reg_val_v = reg_val_df['Pred Probs V'].reset_index().drop(columns = ['index'])
    pred_probs_reg_val_h = reg_val_df['Pred Probs H'].reset_index().drop(columns = ['index'])

    preds_val_reg_arr = []
    pred_probs_reg_val = []
    for i in range(len(preds_val_reg)):
        preds_val_reg_arr.append(preds_val_reg['Prediction'][i])
        pred_probs_reg_val.append([pred_probs_reg_val_v['Pred Probs V'][i], pred_probs_reg_val_h['Pred Probs H'][i]])

    val_reg_odds_v = reg_val_df['V ML'].reset_index().drop(columns = ['index'])
    val_reg_odds_h = reg_val_df['H ML'].reset_index().drop(columns = ['index'])
    
    running_account_reg_val, val_gains_reg = evaluate_model_profit(preds_val_reg_arr, validation_label, val_reg_odds_v, val_reg_odds_h, min_exp_gain, wager_pct, fixed_wager, wager_crit, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_val, scrape = scrape)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Validation Data')
      
    return val_gains_reg

def make_new_bets(current_data_df, class_features, output_label, class_model, reg_features, reg_label, profit_reg_model, reg_threshold, fixed_wager, wager_pct, wager_crit, account):
    """
    Given data for some new games (current_data_df) and the trained layered model, 
    output the games that we should bet on and how much money to bet
    """
    
    #Format the data
    data_class_features = current_data_df[class_features]
    #data_label = current_data_df[output_label]
    data_odds_v = current_data_df['V ML']
    data_odds_h = current_data_df['H ML']
    
    pred_class = class_model.predict(data_class_features) 
    pred_probs_class = class_model.predict_proba(data_class_features)
    
    d_val = {'Pred Probs V': pred_probs_class[:,0], 'Pred Probs H': pred_probs_class[:,1], 'Prediction': pred_class}
    reg_data_df = pd.DataFrame(data=d_val)
    
    reg_data_df['V ML'] = data_odds_v.reset_index().drop(columns = ['index'])
    reg_data_df['H ML'] = data_odds_h.reset_index().drop(columns = ['index'])
    
    reg_data_features = reg_data_df[reg_features]
    expected_profit_data = profit_reg_model.predict(reg_data_features)

    bet_placed_index_store = []
    wager_store = []
    numGames = len(expected_profit_data)
    for i in range(numGames):
        #By default, bet on the game (ie threshold_met = True)
        threshold_met = True
        #Get the expected profit from the regression model and check if it's above the threshold
        exp_gain = expected_profit_data[i]
        if exp_gain < reg_threshold:
            #If the threshold is not met, do not bet on the game
            threshold_met = False
        if threshold_met == True:
            if fixed_wager == False:
                if wager_crit == 'log':
                    wager = wager_pct*account*np.log(exp_gain/4+1.0001)
                if wager_crit == 'kelly':
                    #Could use these numbers as an initial value and continue to track & update them
                    #win_pct = wins/(wins+losses)
                    #w_l_ratio = avg_win/avg_loss
                    win_pct = 0.56
                    w_l_ratio = 1.28
                    k_pct = win_pct - (1-win_pct)/w_l_ratio #0.22
                    wager = k_pct*account #could further multiply this by some f(exp_gain) 
                if wager_crit == 'sqrt':
                    wager = wager_pct*account*np.sqrt(exp_gain/5)
                    
                if wager > 20:
                    wager = 20
            else:
                wager = 10
            
            bet_placed_index_store.append(i)
            wager_store.append(wager)
        else:
            wager_store.append(0)
            
    num_bets_placed = len(bet_placed_index_store)
    print(num_bets_placed, 'bets recommended out of', numGames, 'total games')
    
    return bet_placed_index_store, wager_store, pred_class
    
    
    