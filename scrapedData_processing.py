#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:29:32 2021

@author: noahlefrancois
"""

import pandas as pd

import NBApredFuncs as pf

drop_columns = ['away_minutes_played','away_points','away_losses','date','home_minutes_played','home_points','home_wins','location','losing_name','winner','winning_name','losing_abbr','winning_abbr']
stats_df = pd.read_csv('Data/scraped_boxScore_2017.csv')
stats_df = stats_df.drop(columns = drop_columns)

#Could add the moneyline odds to the df here so the dropped games don't complicate things later
odds_columns = ['Date','VH', 'Team', 'Final', 'ML']
odds_df2015 = pd.read_csv('Data/nba_odds_1516_scrape.csv')[odds_columns]
odds_df2016 = pd.read_csv('Data/nba_odds_1617_scrape.csv')[odds_columns]
odds_df2017 = pd.read_csv('Data/nba_odds_1718_scrape.csv')[odds_columns]
odds_df2020 = pd.read_csv('Data/nba_odds_2021_scrape.csv')[odds_columns]

#Add year columns and concatenate the odds dfs
year2015 = [2015]*len(odds_df2015)
year2016 = [2016]*len(odds_df2016)
year2017 = [2017]*len(odds_df2017)
year2020 = [2020]*len(odds_df2020)
odds_df2015['Year'] = year2015
odds_df2016['Year'] = year2016
odds_df2017['Year'] = year2017
odds_df2020['Year'] = year2020

odds_df = pd.concat([odds_df2015, odds_df2016, odds_df2017, odds_df2020])

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
        
num_Games = len(stats_df)


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

away_pace = stats_df['pace']
home_pace = stats_df['pace']

stats_df['away_pace'] = away_pace
stats_df['home_pace'] = home_pace

#Specify the features we want to include and use for rolling averages
team_list = stats_df.home_abbr.unique().tolist()

away_features = ['away_pace','away_assist_percentage','away_assists','away_block_percentage','away_blocks','away_defensive_rating','away_defensive_rebound_percentage','away_defensive_rebounds','away_effective_field_goal_percentage','away_field_goal_attempts','away_field_goal_percentage','away_field_goals','away_free_throw_attempts','away_free_throw_percentage','away_free_throws','away_offensive_rating','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_three_point_field_goals','away_total_rebound_percentage','away_total_rebounds','away_true_shooting_percentage','away_turnover_percentage','away_turnovers','away_two_point_field_goal_attempts','away_two_point_field_goal_percentage','away_two_point_field_goals']        
home_features = ['home_pace','home_assist_percentage','home_assists','home_block_percentage','home_blocks','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_effective_field_goal_percentage','home_field_goal_attempts','home_field_goal_percentage','home_field_goals','home_free_throw_attempts','home_free_throw_percentage','home_free_throws','home_offensive_rating','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_three_point_field_goals','home_total_rebound_percentage','home_total_rebounds','home_true_shooting_percentage','home_turnover_percentage','home_turnovers','home_two_point_field_goal_attempts','home_two_point_field_goal_percentage','home_two_point_field_goals']
total_features = ['gmDate','away_abbr','home_abbr', 'Season'] + away_features + home_features + ['away_score','home_score']

stats_df = stats_df[total_features]

#Get the winner of each game from the score
winner = []
for i in range(num_Games):
    if stats_df['away_score'][i] > stats_df['home_score'][i]:
        winner.append('V')
    else:
        winner.append('H')

stats_df['Winner'] = winner

#Get the desired season
stats_df2017 = stats_df.loc[stats_df['Season'] == 2017]

#Get the odds for 2015 games. Clearly there's a problem here bc I'm missing 451/1230 2015 games
v_odds_list_2017, h_odds_list_2017, v_score_list_2017, h_score_list_2017, broke_count_2017 = pf.get_season_odds_matched_scrape(stats_df2017, odds_df)
#Add the odds as two new columns in stats_df
stats_df2017['V ML'] = v_odds_list_2017
stats_df2017['H ML'] = h_odds_list_2017

stats_df2017 = stats_df2017[stats_df2017['V ML'] != 0]
#Get the rolling averages for our seasons of interest
prev_num_games = 3

window = 'flat' #options are 'flat' or 'gaussian'
avg = 'rolling' #options are 'rolling' or 'season'
if avg == 'rolling':
    stats_df2017 = pf.avg_previous_num_games(stats_df2017, prev_num_games, window, home_features, away_features, team_list, scrape=True)
if avg == 'season':
    stats_df2017 = pf.avg_season(stats_df2017, home_features, away_features, team_list, scrape=True)


#Need to encode string variables with number labels
team_mapping = dict( zip(team_list,range(len(team_list))) )
stats_df2017.replace({'away_abbr': team_mapping},inplace=True)
stats_df2017.replace({'home_abbr': team_mapping},inplace=True)

#Combine the seasons back into one df
model_data_df = pd.concat([stats_df2017])

#Include the number of games used for averaging as a column in the df
n_val = [prev_num_games]*len(model_data_df)
model_data_df['Num Games Avg'] = n_val

season_list = stats_df['Season'].unique().tolist()
season_mapping = dict( zip(season_list,range(len(season_list))) )
model_data_df.replace({'Season': season_mapping},inplace=True)

#Save the data here and export to another script to run the model
model_data_df.to_csv('Data/pre-processedData_scraped1718_n3.csv', index=False)


