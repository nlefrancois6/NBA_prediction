#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:50:34 2020

@author: noahlefrancois
"""

import pandas as pd

import NBApredFuncs as pf


#Read the data for historical stats
#stats_columns = ['gmDate', 'teamAbbr', 'teamLoc', 'teamRslt', 'teamDayOff', 'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%']
drop_columns = ['offLNm1','offFNm1','offLNm2','offFNm2','offLNm3','offFNm3','teamConf','teamDiv','teamMin','opptConf','opptDiv','opptMin']
stats_df = pd.read_csv('2012-18_teamBoxScore.csv')
stats_df = stats_df.drop(columns = drop_columns)

#Could add the moneyline odds to the df here so the dropped games don't complicate things later
odds_columns = ['Date','VH', 'Team', 'Final', 'ML']
odds_df2015 = pd.read_csv('nba_odds_1516.csv')[odds_columns]
odds_df2016 = pd.read_csv('nba_odds_1617.csv')[odds_columns]
odds_df2017 = pd.read_csv('nba_odds_1718.csv')[odds_columns]

#Add year columns and concatenate the odds dfs
year2015 = [2015]*len(odds_df2015)
year2016 = [2016]*len(odds_df2016)
year2017 = [2017]*len(odds_df2017)
odds_df2015['Year'] = year2015
odds_df2016['Year'] = year2016
odds_df2017['Year'] = year2017

odds_df = pd.concat([odds_df2015, odds_df2016, odds_df2017])


#Need to remove playoff games

#Get the V and H ML values for each pair of rows and assign them to the row that will be kept
#Do the same with the V and H score
ML_V = []
ML_H = []
Score_V = []
Score_H = []
for i in range(len(odds_df)):
    if i%2 == 0:
        ML_V.append(odds_df['ML'].iloc[i])
        ML_H.append(odds_df['ML'].iloc[i+1])
        Score_V.append(odds_df['Final'].iloc[i])
        Score_H.append(odds_df['Final'].iloc[i+1])
    if i%2 == 1:
        ML_V.append(0)
        ML_H.append(0)
        Score_V.append(0)
        Score_H.append(0)
"""
stats_df['ML V'] = ML_V
stats_df['ML H'] = ML_H
"""

num_Games_double = len(stats_df)
num_Games = int(num_Games_double/2)

#Drop every second row since it's duplicated
indexToDrop = []
for i in range(num_Games_double):
    if i%2 == 1:
        indexToDrop.append(i)
        
stats_df = stats_df.drop(index=indexToDrop).reset_index().drop(columns = ['index'])

#Categorize the games by season and add it as a column to stats_df
season = [0]*num_Games
for i in range(num_Games):   
    #If game is in first half of the year, assign it to previous season
    #Playoffs begin ~April 15 every year. Could cut off at april, only lose 2 weeks of games
    #shouldn't be necessary since I only record odds from games that are in stats_df
    if int(stats_df['gmDate'][i][5:7]) < 6:
        season[i] = int(stats_df['gmDate'][i][0:4])-1
    #If game is in second half of the year, assign it to this year's season
    else:
        season[i] = int(stats_df['gmDate'][i][0:4]) 

stats_df['Season'] = season

#Specify the features we want to include and use for rolling averages
team_list = stats_df.teamAbbr.unique().tolist()
#Note that team denotes visitors, oppt denotes home
away_features = ['teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
home_features = ['opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']

total_features = ['teamAbbr','opptAbbr','teamRslt','gmDate','Season','teamDayOff', 'opptDayOff'] + away_features + home_features
stats_df = stats_df[total_features]

#Get the 2015 and 2016 seasons
stats_df2015 = stats_df.loc[stats_df['Season'] == 2015]
stats_df2016 = stats_df.loc[stats_df['Season'] == 2016]
stats_df2017 = stats_df.loc[stats_df['Season'] == 2017]

#Get the rolling averages for our seasons of interest
prev_num_games = 3
window = 'flat' #options are 'flat' or 'gaussian'
avg = 'season' #options are 'rolling' or 'season'
if avg == 'rolling':
    stats_df2015 = pf.avg_previous_num_games(stats_df2015, prev_num_games, window, home_features, away_features, team_list)
    stats_df2016 = pf.avg_previous_num_games(stats_df2016, prev_num_games, window, home_features, away_features, team_list)
    stats_df2017 = pf.avg_previous_num_games(stats_df2017, prev_num_games, window, home_features, away_features, team_list)
if avg == 'season':
    stats_df2015 = pf.avg_season(stats_df2015, home_features, away_features, team_list)
    stats_df2016 = pf.avg_season(stats_df2016, home_features, away_features, team_list)
    stats_df2017 = pf.avg_season(stats_df2017, home_features, away_features, team_list)

#Get the odds for 2015 games
v_odds_list_2015, h_odds_list_2015, v_score_list_2015, h_score_list_2015, broke_count_2015 = pf.get_season_odds_matched(stats_df2015, odds_df)
#Add the odds as two new columns in stats_df, do the same with the score
stats_df2015['V ML'] = v_odds_list_2015
stats_df2015['H ML'] = h_odds_list_2015
stats_df2015['V Score'] = v_score_list_2015
stats_df2015['H Score'] = h_score_list_2015

#Get the odds for 2016 games
v_odds_list_2016, h_odds_list_2016, v_score_list_2016, h_score_list_2016, broke_count_2016 = pf.get_season_odds_matched(stats_df2016, odds_df)
#Add the odds as two new columns in stats_df, do the same with the score
stats_df2016['V ML'] = v_odds_list_2016
stats_df2016['H ML'] = h_odds_list_2016
stats_df2016['V Score'] = v_score_list_2016
stats_df2016['H Score'] = h_score_list_2016

#Get the odds for 2017 games
v_odds_list_2017, h_odds_list_2017, v_score_list_2017, h_score_list_2017, broke_count_2017 = pf.get_season_odds_matched(stats_df2017, odds_df)
#Add the odds as two new columns in stats_df, do the same with the score
stats_df2017['V ML'] = v_odds_list_2017
stats_df2017['H ML'] = h_odds_list_2017
stats_df2017['V Score'] = v_score_list_2017
stats_df2017['H Score'] = h_score_list_2017


#Need to encode string variables with number labels
team_mapping = dict( zip(team_list,range(len(team_list))) )
stats_df2015.replace({'teamAbbr': team_mapping},inplace=True)
stats_df2015.replace({'opptAbbr': team_mapping},inplace=True)
stats_df2016.replace({'teamAbbr': team_mapping},inplace=True)
stats_df2016.replace({'opptAbbr': team_mapping},inplace=True)
stats_df2017.replace({'teamAbbr': team_mapping},inplace=True)
stats_df2017.replace({'opptAbbr': team_mapping},inplace=True)

#Combine the seasons back into one df
model_data_df = pd.concat([stats_df2015, stats_df2016, stats_df2017])

#Include the number of games used for averaging as a column in the df
n_val = [prev_num_games]*len(model_data_df)
model_data_df['Num Games Avg'] = n_val

season_list = stats_df['Season'].unique().tolist()
season_mapping = dict( zip(season_list,range(len(season_list))) )
model_data_df.replace({'Season': season_mapping},inplace=True)

#Save the data here and export to another script to run the model
model_data_df.to_csv('pre-processedData_seasonAvg_n15.csv', index=False)