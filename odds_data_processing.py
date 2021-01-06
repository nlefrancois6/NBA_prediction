#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:04:30 2020

@author: noahlefrancois
"""
import pandas as pd

odds_columns = ['Date','VH', 'Team', 'Final', 'ML']
odds_df = pd.read_csv('Data/nba_odds_1718_scrape.csv')[odds_columns]

#Relabel the team column of odds_df so it matches the format of stats_df
"""
#for Kaggle data
team_dict = {'Atlanta':'ATL', 'Boston':'BOS', 'Brooklyn':'BKN','Charlotte':'CHA','Chicago':'CHI','Cleveland':'CLE','Dallas':'DAL', 'Denver':'DEN','Detroit':'DET',
             'GoldenState':'GS', 'Houston':'HOU', 'Indiana':'IND','LAClippers':'LAC','LA Clippers':'LAC','LALakers':'LAL','Memphis':'MEM','Miami':'MIA','Milwaukee':'MIL','Minnesota':'MIN',
             'NewOrleans':'NO','NewYork':'NY','OklahomaCity':'OKC','Oklahoma City':'OKC','Orlando':'ORL','Philadelphia':'PHI','Phoenix':'PHO','Portland':'POR','Sacramento':'SAC',
             'SanAntonio':'SA','Toronto':'TOR','Utah':'UTA','Washington':'WAS'}
"""
#For scraped data
team_dict = {'Atlanta':'ATL', 'Boston':'BOS', 'Brooklyn':'BRK','Charlotte':'CHO','Chicago':'CHI','Cleveland':'CLE','Dallas':'DAL', 'Denver':'DEN','Detroit':'DET',
             'GoldenState':'GSW', 'Houston':'HOU', 'Indiana':'IND','LAClippers':'LAC','LA Clippers':'LAC','LALakers':'LAL','Memphis':'MEM','Miami':'MIA','Milwaukee':'MIL','Minnesota':'MIN',
             'NewOrleans':'NOP','NewYork':'NYK','OklahomaCity':'OKC','Oklahoma City':'OKC','Orlando':'ORL','Philadelphia':'PHI','Phoenix':'PHO','Portland':'POR','Sacramento':'SAC',
             'SanAntonio':'SAS','Toronto':'TOR','Utah':'UTA','Washington':'WAS'}

for i in range(len(odds_df)):
    odds_df['Team'][i] = team_dict[odds_df['Team'][i]]
    
#Save the relabelled odds_df back to the file so we can load it into NBApred_preprocessing
odds_df.to_csv('Data/nba_odds_1718_scrape.csv', index=False)
    
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
