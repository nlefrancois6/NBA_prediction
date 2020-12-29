#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd
import numpy as np
from sklearn import ensemble, metrics
import matplotlib.pyplot as plt

#Load the pre-processed data
model_data_df = pd.read_csv("pre-processedData_n5.csv")

#Select the model features
away_features = ['teamFG%','teamEFG%','teamOrtg','teamEDiff']
home_features = ['opptTS%','opptEFG%','opptPPS','opptDrtg','opptEDiff','opptAST/TO','opptSTL/TO']
features = ['teamAbbr','opptAbbr','Season'] + away_features + home_features
output_label = ['teamRslt'] 
#Note teamRslt = Win means visitors win, teamRslt = Loss means home wins

#Separate training and testing set
training_df = model_data_df.sample(frac=0.8, random_state=1)
indlist=list(training_df.index.values)

testing_df = model_data_df.copy().drop(index=indlist)

#Define the features (input) and label (prediction output) for training set
training_features = training_df[features]
training_label = training_df[output_label]

#Define features and label for testing set
testing_features = testing_df[features]
testing_label = testing_df[output_label]

#Train a Gradient Boosting Machine on the data
gbc = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.02, max_depth=1)
gbc.fit(training_features, training_label)

#Predict the outcome from our test set and evaluate the prediction accuracy for each model
predGB = gbc.predict(testing_features) 
pred_probsGB = gbc.predict_proba(testing_features) #probability of [results==True, results==False]
accuracyGB = metrics.accuracy_score(testing_label, predGB)


plot_features = False
if plot_features:
    #Plot feature importances
    feature_importance = gbc.feature_importances_.tolist()
    f2=plt.figure()
    plt.bar(features,feature_importance)
    plt.title("Gradient Boosting Classifier: Feature Importance")
    plt.xticks(rotation='vertical')
    plt.show()


#Could filter the testing set to only "bet" on games where we meet a minimum confidence
#Will need to check whether this raises or lowers profit since we might be missing upsets and actually lose money since we only bet on strong favourites
min_conf_filter = False
if min_conf_filter:
    min_confidence = 0.7

    predGB_minConfSatisfied = []
    predGB_label_minConfSatisfied = []

    labels_arr = testing_label['teamRslt'].array

    for i in range(len(predGB)):
        if (pred_probsGB[i,0] > min_confidence) or (pred_probsGB[i,1] > min_confidence):
            predGB_minConfSatisfied.append(predGB[i])
            predGB_label_minConfSatisfied.append(labels_arr[i])

    accuracy_minConf = metrics.accuracy_score(predGB_label_minConfSatisfied, predGB_minConfSatisfied)

def calc_Profit(account, wager_pct, winner_prediction, winner_actual, moneyline_odds):
    """
    account: total money in the account at the start
    
    wager_pct: the amount wagered on each game as a fraction of the account. 
        float [0,1]
    
    winner_prediction: the prediction of whether visiting team will win or lose.
        Possible values are 'Win' and 'Loss'
    
    winner_actual: the actual result of whether visiting team won or lost.
        Possible values are 'Win' and 'Loss' (might need to handle 'push')
    
    moneyline_odds: the moneyline odds given for visiting & home teams
        Not sure of format yet but probably (numGames,2) array with [V odds, H odds]
        Might need to apply a conversion for negative (ie favourite) odds, or handle the negative here
    
    Returns account_runningTotal, an array containing the total money we have after each game
    """
    
    account_runningTotal = [account]
    gain = 0
    numGames = len(winner_prediction)
    for i in range(numGames):
        wager = wager_pct*account
        #If our prediction was correct, calculate the winnings
        if winner_actual[i] == winner_prediction[i]:
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
            gain = -wager
        
        account = account + gain
        account_runningTotal.append(account)
        
    return account_runningTotal

profit_calc = True
if profit_calc:
    account = 100
    wager_pct = 0.1
    winner_prediction = predGB
    winner_actual = testing_label['teamRslt'].array
    moneyline_odds = np.zeros([len(winner_prediction),2])
    for i in range(len(winner_prediction)):
        moneyline_odds[i,0] = 120
        moneyline_odds[i,1] = -140
    account_runningTotal = calc_Profit(account, wager_pct, winner_prediction, winner_actual, moneyline_odds)
    
    plt.figure()
    plt.plot(account_runningTotal)
    plt.hlines(account, 0, len(account_runningTotal),linestyles='dashed')
    plt.xlabel('Games')
    plt.ylabel('Account Total')
    plt.title('Betting Performance of Our Model')
    