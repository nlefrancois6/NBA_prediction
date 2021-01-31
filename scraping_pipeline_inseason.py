#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 00:03:03 2021

@author: noahlefrancois
"""

import pandas as pd
from datetime import datetime
from sportsreference.nba.boxscore import Boxscores

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import NBApredFuncs as pf

#Set the date label for scraped games
gmDate_odds = 130
gmDate_boxscore = '2021-01-30'
year_num = 2020

#Switch to update csv files
save_games = False

#Scrape the new games
update_games = False
if update_games:
    games = Boxscores(datetime(2021, 1, 30), datetime(2021, 1, 31))
    schedule_dict = games.games 
    #each entry in the dict is another dict containing all the games from a single day
    #Need to unpack this into a single dict before I can store it in a df
    year = 2021
    day = 30
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
    
    SBR_scrape = False
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
        odds_df = pd.read_csv('Data/nba_odds_2021_injTest.csv')[odds_columns]
        
    injury_scrape = True
    if injury_scrape:
        abbr_list = ['ATL', 'BOS', 'BRK','CHO','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
        
        inj_today = pf.scrape_todays_injury_report(gmDate_boxscore)
        
        load_stats = True
        if load_stats:
            league_players_df = pd.read_csv('Data/leaguePlayersStats_current.csv')
            team_totals_df = pd.read_csv('Data/teamTotalStats_current.csv')   
        else:
            #Get dfs with the statistical contributions of all players and all summed rosters for the season
            league_players_df, team_totals_df = pf.get_stat_contributions(abbr_list, year_num, gmDate_boxscore)
        
        #save team and league stats to a csv
        save_stats = False
        if save_stats:
            league_players_df.to_csv('Data/leaguePlayersStats_current.csv', index=False)
            team_totals_df.to_csv('Data/teamTotalStats_current.csv', index=False)
        
        #Given today's injury report, get the fraction of each team's total statistics which are missing due to injury
        team_injLoss_df = pf.get_injLoss_daily_report(inj_today, league_players_df, team_totals_df, abbr_list)
        
        #Add today's injury losses to the running list for the season
        team_injLoss_Year_df = pd.read_csv('Data/injuryLosses2020.csv')
        inj_df= pd.concat([team_injLoss_Year_df, team_injLoss_df])
             
        if save_games:
            inj_df.to_csv('Data/injuryLosses2020.csv', index=False)
        
        #Merge team statistics and injury loss dfs
        #game_df = pf.preprocess_scraped_data(game_df, odds_df, inj_df, away_features, home_features, total_features, prev_num_games, encode_teams, year_int)
            
    #Select the features we want to use in our model
    away_features = ['away_pace','away_assist_percentage','away_assists','away_block_percentage','away_blocks','away_defensive_rating','away_defensive_rebound_percentage','away_defensive_rebounds','away_effective_field_goal_percentage','away_field_goal_attempts','away_field_goal_percentage','away_field_goals','away_free_throw_attempts','away_free_throw_percentage','away_free_throws','away_offensive_rating','away_offensive_rebounds','away_personal_fouls','away_steal_percentage','away_steals','away_three_point_attempt_rate','away_three_point_field_goal_attempts','away_three_point_field_goal_percentage','away_three_point_field_goals','away_total_rebound_percentage','away_total_rebounds','away_true_shooting_percentage','away_turnover_percentage','away_turnovers','away_two_point_field_goal_attempts','away_two_point_field_goal_percentage','away_two_point_field_goals']        
    home_features = ['home_pace','home_assist_percentage','home_assists','home_block_percentage','home_blocks','home_defensive_rating','home_defensive_rebound_percentage','home_defensive_rebounds','home_effective_field_goal_percentage','home_field_goal_attempts','home_field_goal_percentage','home_field_goals','home_free_throw_attempts','home_free_throw_percentage','home_free_throws','home_offensive_rating','home_offensive_rebounds','home_personal_fouls','home_steal_percentage','home_steals','home_three_point_attempt_rate','home_three_point_field_goal_attempts','home_three_point_field_goal_percentage','home_three_point_field_goals','home_total_rebound_percentage','home_total_rebounds','home_true_shooting_percentage','home_turnover_percentage','home_turnovers','home_two_point_field_goal_attempts','home_two_point_field_goal_percentage','home_two_point_field_goals']
    total_features = ['gmDate','away_abbr','home_abbr', 'Season'] + away_features + home_features + ['away_score','home_score']
            
    year_int = 2020
    prev_num_games = 3
    encode_teams = False
    #Perform preprocessing to get data that is ready to be used in the model
    print('Merging odds & stats data, processing final dataframe...')
    game_df = pf.preprocess_scraped_data(game_df, odds_df, inj_df, away_features, home_features, total_features, prev_num_games, encode_teams, year_int)
    print('Preprocessing completed')
    #Save the data here and export to another script to run the model
    if save_games:
        game_df.to_csv('Data/pre-processedData_scraped2021_n3_inj.csv', index=False)
        print('Updated data saved')
    else:
        print('Updated data not saved')
