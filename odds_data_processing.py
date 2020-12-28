#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:04:30 2020

@author: noahlefrancois
"""
import pandas as pd

"""
#Read the data for historical odds
odds_columns = ['Date','VH', 'Team', 'Final', 'ML']
odds_df = pd.read_csv('nba_odds_1819.csv')[odds_columns]
num_Games_double = len(odds_df)
num_Games = int(num_Games_double/2)

#get predicted spread or predicted winner:
ML_odds = odds_df['ML'] < 0 #boolean array, 1:pred winner, 0:pred loser

#Get the actual winner
ML_actual = [0]*num_Games_double
first = 0
second = 1
for i in range(num_Games):
    if odds_df['Final'][second] > odds_df['Final'][first]:
        ML_actual[second] = 1
    else:
        ML_actual[first] = 1
    first = first + 2
    second = second + 2

#Store the ML prediction winner and the actual winner
odds_df['Pred Winner'] = ML_odds
odds_df['Actual Winner'] = ML_actual

#Store a column that records whether the ML prediction was correct (1) or incorrect (0)
ML_correct = ML_odds == ML_actual
odds_df['ML Correct'] = ML_correct
"""

def calc_Profit(wager, winner_prediction, winner_actual, moneyline_odds):
    """
    wager: the amount wagered on each game. Assume this is the same for all games.
    
    winner_prediction: the prediction of whether visiting team will win or lose.
        Possible values are 'Win' and 'Loss'
    
    winner_actual: the actual result of whether visiting team won or lost.
        Possible values are 'Win' and 'Loss' (might need to handle 'push')
    
    moneyline_odds: the moneyline odds given for visiting & home teams
        Not sure of format yet but probably (numGames,2) array with [V odds, H odds]
        Might need to apply a conversion for negative (ie favourite) odds, or handle the negative here
    
    Returns gain
    """
    gain = 0
    numGames = len(winner_prediction)
    for i in numGames:
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
                    gain = moneyline_odds[1]*(wager/100)
                else:
                    gain = 100*(wager/(-moneyline_odds[i,1]))
        #If our prediction was wrong, lose the wager
        else:
            gain = -wager
            
    return gain
        
        
        
        