#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 00:03:03 2021

@author: noahlefrancois
"""

import pandas as pd
from datetime import datetime
from sportsreference.nba.boxscore import Boxscore, Boxscores

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import NBApredFuncs as pf

#Set the date label for scraped games
gmDate_odds = 119
gmDate_boxscore = '2021-01-19'

#Switch to update csv files
save_games = False

#Scrape the new games
update_games = False
if update_games:
    games = Boxscores(datetime(2021, 1, 18), datetime(2021, 1, 19))
    schedule_dict = games.games 
    #each entry in the dict is another dict containing all the games from a single day
    #Need to unpack this into a single dict before I can store it in a df
    year = 2021
    day = 18
    month = 1
    numDays = len(schedule_dict)
    
    #Scrape the boxscores for each game in schedule_dict and save them all to a df
    game_df = pf.scrape_boxScore(schedule_dict, year, month, day, numDays)
    
    #Option to load data we already have from this year and just append the new data
    update_year = True
    if update_year:
        old_game_df = pd.read_csv('Data/scraped_boxScore_2020.csv')
        game_df = pd.concat([old_game_df, game_df])

    if save_games:
        game_df.to_csv('Data/scraped_boxScore_2020.csv', index=False)
else:
    game_df = pd.read_csv('Data/scraped_boxScore_2020.csv')

preprocess = True
if preprocess:
    #Process the scraped data & odds data
    drop_columns = ['away_minutes_played','away_points','away_losses','date','home_minutes_played','home_points','home_wins','location','losing_name','winner','winning_name','losing_abbr','winning_abbr']
    game_df = game_df.drop(columns = drop_columns).reset_index().drop(columns = ['index'])
    
    SBR_scrape = True
    if SBR_scrape:
        verbose = False
        
        filename = 'odds_data_SBR.csv'
        pf.scrape_SBR_odds(filename)
        todays_odds = pd.read_csv(filename)
        
        #Get odds_df_twoRows, which contains the odds for today in the proper format
        odds_df_oneRow, odds_df_twoRows = pf.reformat_scraped_odds(todays_odds, gmDate_odds, verbose)
        #Load the existing scraped odds list
        odds_columns = ['Date','VH', 'Team', 'Final','ML']
        odds_old_df = pd.read_csv('Data/nba_odds_2021_scrape.csv')[odds_columns]
        #Add today's odds to the list and save the updated list
        odds_combined_df = pd.concat([odds_old_df, odds_df_twoRows]).reset_index().drop(columns = ['index'])
        if save_games:
            odds_combined_df.to_csv('Data/nba_odds_2021_scrape.csv')
        
        odds_df = odds_combined_df.copy()
        

        
        game_df, game_df_newRows = pf.concat_upcoming_games(game_df, odds_df_oneRow, gmDate_boxscore)
    else:
        odds_columns = ['Date','VH', 'Team', 'Final','ML']
        odds_df = pd.read_csv('Data/nba_odds_2021_scrape.csv')[odds_columns]
    """
    https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/?date=20200106
    Need to manually get the odds from this url for each date (YYYYMMDD format) to fill in the
    ones that aren't updated yet from the historical data. The historical data page claims they
    try to update it ~weekly so it shouldn't be missing too many games; but it will never have
    the odds from a game that hasn't been played yet.
    """
    
    away_features = ['away_pace','away_assist_percentage','away_assists','away_block_percentage','away_blocks','away_defensive_rating','away_defensive_rebound_percentage','away_defensive_rebounds','away_effective_field_goal_percentage','away_field_goal_attempts','away_field_goal_percentage','away_field_goals','away_free_throw_attempts','away_free_throw_percentage','away_free_throws','away_offensive_rating','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_three_point_field_goals','away_total_rebound_percentage','away_total_rebounds','away_true_shooting_percentage','away_turnover_percentage','away_turnovers','away_two_point_field_goal_attempts','away_two_point_field_goal_percentage','away_two_point_field_goals']        
    home_features = ['home_pace','home_assist_percentage','home_assists','home_block_percentage','home_blocks','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_effective_field_goal_percentage','home_field_goal_attempts','home_field_goal_percentage','home_field_goals','home_free_throw_attempts','home_free_throw_percentage','home_free_throws','home_offensive_rating','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_three_point_field_goals','home_total_rebound_percentage','home_total_rebounds','home_true_shooting_percentage','home_turnover_percentage','home_turnovers','home_two_point_field_goal_attempts','home_two_point_field_goal_percentage','home_two_point_field_goals']
    total_features = ['gmDate','away_abbr','home_abbr', 'Season'] + away_features + home_features + ['away_score','home_score']
            
    year_int = 2020
    prev_num_games = 3
    encode_teams = True
    
    game_df = pf.preprocess_scraped_data(game_df, odds_df, away_features, home_features, total_features, prev_num_games, encode_teams, year_int)
    """
    #make a func: game_df = preprocess_scraped_data(game_df, odds_df, away_features, home_features, total_features, encode_teams)
    #Add year columns and concatenate the odds dfs
    year = [2020]*len(odds_df)
    odds_df['Year'] = year
    
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
            
    num_Games = len(game_df)
    
    #Categorize the games by season and add it as a column to stats_df
    season = [0]*num_Games
    for i in range(num_Games):   
        #If game is in first half of the year, assign it to previous season
        if int(game_df['gmDate'][i][5:7]) < 6:
            season[i] = int(game_df['gmDate'][i][0:4])-1
        #If game is in second half of the year, assign it to this year's season
        else:
            season[i] = int(game_df['gmDate'][i][0:4]) 
    
    game_df['Season'] = season
    
    away_pace = game_df['pace']
    home_pace = game_df['pace']
    
    game_df['away_pace'] = away_pace
    game_df['home_pace'] = home_pace
    
    #Specify the features we want to include and use for rolling averages
    team_list = game_df.home_abbr.unique().tolist()
    
    away_features = ['away_pace','away_assist_percentage','away_assists','away_block_percentage','away_blocks','away_defensive_rating','away_defensive_rebound_percentage','away_defensive_rebounds','away_effective_field_goal_percentage','away_field_goal_attempts','away_field_goal_percentage','away_field_goals','away_free_throw_attempts','away_free_throw_percentage','away_free_throws','away_offensive_rating','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_three_point_field_goals','away_total_rebound_percentage','away_total_rebounds','away_true_shooting_percentage','away_turnover_percentage','away_turnovers','away_two_point_field_goal_attempts','away_two_point_field_goal_percentage','away_two_point_field_goals']        
    home_features = ['home_pace','home_assist_percentage','home_assists','home_block_percentage','home_blocks','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_effective_field_goal_percentage','home_field_goal_attempts','home_field_goal_percentage','home_field_goals','home_free_throw_attempts','home_free_throw_percentage','home_free_throws','home_offensive_rating','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_three_point_field_goals','home_total_rebound_percentage','home_total_rebounds','home_true_shooting_percentage','home_turnover_percentage','home_turnovers','home_two_point_field_goal_attempts','home_two_point_field_goal_percentage','home_two_point_field_goals']
    total_features = ['gmDate','away_abbr','home_abbr', 'Season'] + away_features + home_features + ['away_score','home_score']
    
    game_df = game_df[total_features]
    
    #Get the winner of each game from the score
    winner = []
    for i in range(num_Games):
        if game_df['away_score'][i] > game_df['home_score'][i]:
            winner.append('V')
        else:
            winner.append('H')
    
    game_df['Winner'] = winner
    
    #Get the odds for each game. Clearly there's a problem here bc I'm missing 451/1230 2015 games
    v_odds_list, h_odds_list, v_score_list, h_score_list, broke_count = pf.get_season_odds_matched_scrape(game_df, odds_df)
    #Add the odds as two new columns in stats_df
    game_df['V ML'] = v_odds_list
    game_df['H ML'] = h_odds_list
    
    #We can manually add the odds for today's games
    #game_df = game_df[game_df['V ML'] != 0]
    
    #Get the rolling averages for our seasons of interest
    prev_num_games = 3
    
    window = 'flat' #options are 'flat' or 'gaussian'
    avg = 'rolling' #options are 'rolling' or 'season'
    if avg == 'rolling':
        game_df = pf.avg_previous_num_games(game_df, prev_num_games, window, home_features, away_features, team_list, scrape=True)
    if avg == 'season':
        game_df = pf.avg_season_scrape(game_df, home_features, away_features, team_list, scrape=True)
    
    encode_teams = False
    if encode_teams:
        #Need to encode string variables with number labels
        team_mapping = dict( zip(team_list,range(len(team_list))) )
        game_df.replace({'away_abbr': team_mapping},inplace=True)
        game_df.replace({'home_abbr': team_mapping},inplace=True)
    
    #Include the number of games used for averaging as a column in the df
    n_val = [prev_num_games]*len(game_df)
    game_df['Num Games Avg'] = n_val
    
    season_list = game_df['Season'].unique().tolist()
    season_mapping = dict( zip(season_list,range(len(season_list))) )
    game_df.replace({'Season': season_mapping},inplace=True)
    """
    #Save the data here and export to another script to run the model
    if save_games:
        game_df.to_csv('Data/pre-processedData_scraped2021_n3.csv', index=False)
