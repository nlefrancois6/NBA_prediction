#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:00:25 2021

@author: noahlefrancois
"""

import pandas as pd
import numpy as np
import math

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


todays_odds = pd.read_csv('odds_data_SBR.csv')

verbose = False
#Get the average odds for each team in each game
v_team_list = []
h_team_list = []
v_odds_list = []
h_odds_list = []
ind_nan = []
numGames = int(len(todays_odds)/2)
for i in range(numGames):
    v_odds_row = todays_odds.iloc[2*i]
    h_odds_row = todays_odds.iloc[2*i + 1]
    
    v_team = v_odds_row['team']
    h_team = h_odds_row['team']
    
    
    v_odds = []
    h_odds = []
    
    for s in range(len(v_odds_row)):
        if (type(v_odds_row[s]) is np.float64) and (math.isnan(v_odds_row[s]) is False):
            v_odds.append(v_odds_row[s])
        if type(h_odds_row[s]) is np.float64 and (math.isnan(v_odds_row[s]) is False):
            h_odds.append(h_odds_row[s])
    
    avg_v_odds = np.mean(v_odds)
    avg_h_odds = np.mean(h_odds)
    
    #Check which team is favoured and check for arbitrage opportunities
    if (avg_v_odds < 0) and (avg_h_odds > 0):
        if verbose:
            print(v_team, '(away) favourite. Average odds', avg_v_odds)
            print(h_team, '(home) underdog. Average odds', avg_h_odds)
        
        if np.abs(min(v_odds)) < np.abs(max(h_odds)):
            if verbose:
                print('Arbitrage Opportunity for', v_team,'vs',h_team)
        
    elif (avg_h_odds < 0) and (avg_v_odds > 0):
        if verbose:
            print(h_team, '(home) favourite. Average odds', avg_h_odds)
            print(v_team, '(away) underdog. Average odds', avg_v_odds)
        
        if np.abs(min(h_odds)) < np.abs(max(v_odds)):
            if verbose:
                print('Arbitrage Opportunity for', v_team,'vs',h_team)
    
    v_team_list.append(v_team)
    h_team_list.append(h_team)
    v_odds_list.append(avg_v_odds)
    h_odds_list.append(avg_h_odds)
    
    if math.isnan(avg_v_odds) or math.isnan(avg_h_odds):
        ind_nan.append(i)

d_odds = {'away_abbr': v_team_list, 'home_abbr': h_team_list, 'V ML': v_odds_list, 'H ML': h_odds_list}  
odds_df = pd.DataFrame(data=d_odds)

#Drop games with no odds (postponed, etc)
if len(ind_nan) > 0:
    odds_df = odds_df.drop(ind_nan).reset_index().drop(columns = ['index'])

#Relabel team names to abbreviations
team_dict = {'Atlanta':'ATL', 'Boston':'BOS', 'Brooklyn':'BRK','Charlotte':'CHO','Chicago':'CHI','Cleveland':'CLE','Dallas':'DAL', 'Denver':'DEN','Detroit':'DET',
             'GoldenState':'GSW', 'Houston':'HOU', 'Indiana':'IND','L.A. Clippers':'LAC','L.A. Lakers':'LAL','Memphis':'MEM','Miami':'MIA','Milwaukee':'MIL','Minnesota':'MIN',
             'New Orleans':'NOP','New York':'NYK','Oklahoma City':'OKC','Orlando':'ORL','Philadelphia':'PHI','Phoenix':'PHO','Portland':'POR','Sacramento':'SAC',
             'San Antonio':'SAS','Toronto':'TOR','Utah':'UTA','Washington':'WAS'}

for i in range(len(odds_df)):
    odds_df['away_abbr'][i] = team_dict[odds_df['away_abbr'][i]]
    odds_df['home_abbr'][i] = team_dict[odds_df['home_abbr'][i]]
    
"""
Incorporating this into the inseason scraping pipeline:
Want to load the existing odds data and append today's odds. Can also use this list of games 
add the necessary rows to the scraped box score df.
"""
    
    
    
    