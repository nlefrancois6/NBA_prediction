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
odds_columns = ['Date','VH', 'Team', 'Final', 'ML']
odds_df2015 = pd.read_csv('nba_odds_1516.csv')[odds_columns]
odds_df2016 = pd.read_csv('nba_odds_1617.csv')[odds_columns]

#Add year columns and concatenate the odds dfs
year2015 = [2015]*len(odds_df2015)
year2016 = [2016]*len(odds_df2016)
odds_df2015['Year'] = year2015
odds_df2016['Year'] = year2016

odds_df = pd.concat([odds_df2015, odds_df2016])


#Need to remove playoff games

#Get the V and H ML values for each pair of rows and assign them to the row that will be kept
ML_V = []
ML_H = []
for i in range(len(odds_df)):
    if i%2 == 0:
        ML_V.append(odds_df['ML'].iloc[i])
        ML_H.append(odds_df['ML'].iloc[i+1])
    if i%2 == 1:
        ML_V.append(0)
        ML_H.append(0)
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
away_features = ['teamDayOff','teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
home_features = ['opptDayOff','opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']

total_features = ['teamAbbr','opptAbbr','teamRslt','gmDate','Season'] + away_features + home_features
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

#Get the odds corresponding to every game in stats_df
def get_season_odds_matched(stats_df_year, odds_df):
    """
    Given stats_df for a year, find the corresponding odds for every game and return
    the v and h odds in two arrays ready to be added to the df. Also return the number
    of games that encountered the size==0 error.
    """
    v_odds_list = []
    h_odds_list = []
    broke_count = 0
    for i in range(len(stats_df_year)):
        season = stats_df_year['Season'].iloc[i]
        game_date = stats_df_year['gmDate'].iloc[i]
        v_team = stats_df_year['teamAbbr'].iloc[i]
        h_team = stats_df_year['opptAbbr'].iloc[i]
        v_odds, h_odds = get_matching_game_odds(game_date, season, v_team, h_team, odds_df)
    
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

        v_odds_list.append(v_odds_num)
        h_odds_list.append(h_odds_num)
    
    return v_odds_list, h_odds_list, broke_count

"""
v_odds_list = []
h_odds_list = []
broke_count_2016 = 0
for i in range(len(stats_df2016)):
    season = stats_df2016['Season'].iloc[i]
    game_date = stats_df2016['gmDate'].iloc[i]
    v_team = stats_df2016['teamAbbr'].iloc[i]
    h_team = stats_df2016['opptAbbr'].iloc[i]
    v_odds, h_odds = get_matching_game_odds(game_date, season, v_team, h_team, odds_df)
    
    if (v_odds.size == 0) or (h_odds.size == 0):
        broke_count_2016 = broke_count_2016+ 1
        #Sometimes the odds have size 0. I think this is probably because the game isn't found
        #Maybe I'll want to drop that one, but for now just setting it to a neutral odds
        #should be fine
        v_odds_num = 100
        h_odds_num = 100
    else:
        v_odds_num = v_odds.iloc[0]
        h_odds_num = h_odds.iloc[0]

    v_odds_list.append(v_odds_num)
    h_odds_list.append(h_odds_num)
"""

#Get the odds for 2015 games
v_odds_list_2015, h_odds_list_2015, broke_count_2015 = get_season_odds_matched(stats_df2015, odds_df)
#Add the odds as two new columns in stats_df
stats_df2015['V ML'] = v_odds_list_2015
stats_df2015['H ML'] = h_odds_list_2015

#Get the odds for 2015 games
v_odds_list_2016, h_odds_list_2016, broke_count_2016 = get_season_odds_matched(stats_df2016, odds_df)
#Add the odds as two new columns in stats_df
stats_df2016['V ML'] = v_odds_list_2016
stats_df2016['H ML'] = h_odds_list_2016

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

#Save the data here and export to another script to run the model
#model_data_df.to_csv('pre-processedData_n5.csv', index=False)
