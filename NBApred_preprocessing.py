#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:50:34 2020

@author: noahlefrancois
"""

import pandas as pd


#Read the data for historical stats
#stats_columns = ['gmDate', 'teamAbbr', 'teamLoc', 'teamRslt', 'teamDayOff', 'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%']
drop_columns = ['offLNm1','offFNm1','offLNm2','offFNm2','offLNm3','offFNm3','teamConf','teamDiv','teamMin','opptConf','opptDiv','opptMin']
stats_df = pd.read_csv('2012-18_teamBoxScore.csv')
stats_df = stats_df.drop(columns = drop_columns)

#Could add the moneyline odds to the df here so the dropped games don't complicate things later

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
    if int(stats_df['gmDate'][i][5:7]) < 8:
        season[i] = int(stats_df['gmDate'][i][0:4])-1
    #If game is in second half of the year, assign it to this year's season
    else:
        season[i] = int(stats_df['gmDate'][i][0:4]) 

stats_df['Season'] = season

#Specify the features we want to include and use for rolling averages
team_list = stats_df.teamAbbr.unique().tolist()
#Note that team denotes visitors, oppt denotes home
away_features = ['teamDayOff','teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
home_features = ['opptDayOff','opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']

total_features = ['teamAbbr','teamRslt','opptAbbr','Season'] + away_features + home_features
stats_df = stats_df[total_features]

def avg_previous_num_games(df, num_games):
    # This function changes each stat to be the average of the last num_games for each team, and shifts it one so it does not include the current stats and drops the first num_games that become null
    for col in home_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
    for col in away_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
    return df.dropna()

#Get the 2015 and 2016 seasons
stats_df2015 = stats_df.loc[stats_df['Season'] == 2015]
stats_df2016 = stats_df.loc[stats_df['Season'] == 2016]

#Get the rolling averages for our seasons of interest
prev_num_games = 5
stats_df2015 = avg_previous_num_games(stats_df2015, prev_num_games)
stats_df2016 = avg_previous_num_games(stats_df2016, prev_num_games)

#Need to encode string variables with number labels
team_mapping = dict( zip(team_list,range(len(team_list))) )
stats_df2015.replace({'teamAbbr': team_mapping},inplace=True)
stats_df2015.replace({'opptAbbr': team_mapping},inplace=True)
stats_df2016.replace({'teamAbbr': team_mapping},inplace=True)
stats_df2016.replace({'opptAbbr': team_mapping},inplace=True)

#Combine the seasons back into one df
model_data_df = pd.concat([stats_df2015, stats_df2016])

season_list = stats_df['Season'].unique().tolist()
season_mapping = dict( zip(season_list,range(len(season_list))) )
model_data_df.replace({'Season': season_mapping},inplace=True)

#I should probably save the data here and export to another script to run the model

model_data_df.to_csv('pre-processedData_n5.csv', index=False)
