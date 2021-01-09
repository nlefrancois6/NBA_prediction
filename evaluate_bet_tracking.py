#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:05:48 2021

@author: noahlefrancois
"""

import pandas as pd
import matplotlib.pyplot as plt

#bet_tracking_df = pd.read_csv('bet_tracking2021_originalModel.csv')
bet_tracking_df = pd.read_csv('bet_tracking2021_optModel.csv')


bet_success = []
for i in range(len(bet_tracking_df)):
    if bet_tracking_df['Recommended Winner'].iloc[i] == bet_tracking_df['Winner'].iloc[i]:
        bet_success.append('G')
    else:
        bet_success.append('L')
    
bet_tracking_df['Bet Success'] = bet_success

winner_actual = bet_tracking_df['Winner']
winner_prediction = bet_tracking_df['Recommended Winner']
wagers = bet_tracking_df['Recommended Wager']
v_odds = bet_tracking_df['V ML']
h_odds = bet_tracking_df['H ML']


account = 100
account_runningTotal = [account]
gains_store = []
gain = 0

numGames = len(winner_prediction)
for i in range(numGames):
    wager = wagers[i]
    #If our prediction was correct, calculate the winnings
    if winner_actual[i] == winner_prediction[i]:
        #wins += 1
        if winner_prediction[i] == 'V':
            #odds[0] is odds on visitor win
            if v_odds[i]>0:
                gain = v_odds[i]*(wager/100)
            else:
                gain = 100*(wager/(-v_odds[i]))
        if winner_prediction[i] == 'H':
            #odds[1] is odds on home win
            if h_odds[i]>0:
                gain = h_odds[i]*(wager/100)
            else:
                gain = 100*(wager/(-h_odds[i]))

    #If our prediction was wrong, lose the wager
    else:
        #losses += 1
        gain = -wager
    
    account = account + gain
    account_runningTotal.append(account)
    gains_store.append(gain)

numGames_bet = len(account_runningTotal)
print('Final account balance after ', numGames_bet, ' bets: ', account)

plt.figure()
plt.plot(account_runningTotal[:19])
plt.title('Model Performance To Date (2020-2021 Season)')
plt.xlabel('Number of Games')
plt.ylabel('Account Balance')
plt.hlines(account_runningTotal[0], 0, len(account_runningTotal),linestyles='dashed')


