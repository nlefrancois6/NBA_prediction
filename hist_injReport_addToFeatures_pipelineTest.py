#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:00:01 2021

@author: noahlefrancois
"""

import pandas as pd
import time
import NBApredFuncs as pf
#import math

year_num = 2020
year_str = '2020-21'

dataYear_df = pd.read_csv('Data/pre-processedData_scraped_abbr_2021_n3.csv')
#Replace this with the injury data from this season up to yesterday
team_injLoss_Year_df = pd.read_csv('Data/injuries_historical_dailyReport_2019.csv')
"""
print('Relabelling Team Names...')
team_dict = {'Hawks':'ATL', 'Celtics':'BOS', 'Nets':'BRK','Hornets':'CHO','Bulls':'CHI','Cavaliers':'CLE','Mavericks':'DAL', 'Nuggets':'DEN','Pistons':'DET',
             'Warriors':'GSW', 'Rockets':'HOU', 'Pacers':'IND','Clippers':'LAC','Lakers':'LAL','Grizzlies':'MEM','Heat':'MIA','Bucks':'MIL','Timberwolves':'MIN',
             'Pelicans':'NOP','Knicks':'NYK','Thunder':'OKC','Magic':'ORL','76ers':'PHI','Suns':'PHO','Blazers':'POR','Kings':'SAC',
             'Spurs':'SAS','Raptors':'TOR','Jazz':'UTA','Wizards':'WAS'}

for i in range(len(injYear_df)):
    injYear_df['Team'][i] = team_dict[injYear_df['Team'][i]]
"""
abbr_list = ['ATL', 'BOS', 'BRK','CHO','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
#abbr_list = ['ATL']

#Get and reformat today's injury report
today = '2021-01-21'
inj_today = pf.scrape_todays_injury_report(today)

t1 = time.time()

"""
This is by far the longest part (~20 minutes). Maybe I could just update it once a week or
so, save it to a csv and load it for the daily process.
"""
load_stats = True
if load_stats:
    league_players_df = pd.read_csv('Data/leaguePlayersStats_current.csv')
    team_totals_df = pd.read_csv('Data/teamTotalStats_current.csv')
else:
    #Get dfs with the statistical contributions of all players and all summed rosters for the season
    league_players_df, team_totals_df = pf.get_stat_contributions(abbr_list, year_num, year_str)
    
t2 = time.time()
t_scrape = (t2-t1)/60
print(t_scrape, 'minutes to scrape. Scraping completed.')

#save team and league stats to a csv
save_stats = False
if save_stats:
    league_players_df.to_csv('Data/leaguePlayersStats_current.csv', index=False)
    team_totals_df.to_csv('Data/teamTotalStats_current.csv', index=False)


#Given today's injury report, get the fraction of each team's total statistics which are missing due to injury
team_injLoss_df = pf.get_injLoss_daily_report(inj_today, league_players_df, team_totals_df, abbr_list)

t3 = time.time()
t_process = (t3-t2)/60
print(t_process, 'minutes to process daily injury report features')

#Add today's injury losses to the running list for the season
injYear_df_combined = pd.concat([team_injLoss_Year_df, team_injLoss_df])

#save team_injLoss_df to a csv
save_injLoss = False
if save_injLoss:
    injYear_df_combined.to_csv('Data/injuryLosses2020.csv', index=False)

#Merge team statistics and injury loss dfs
#dataYear_merged_df = pf.merge_injury_data(dataYear_df, team_injLoss_df)
"""
Problem: this gives an error if there are days without injury report (ie earlier in the
season when I wasn't tracking it). 
Planned Solution: I need to have the injury report for all the days before today 
stored, load it, and append today's injuries to it. Then I'll save it after to use the next day.
"""

#Encode the team abbreviations back to numerical labels
team_list = dataYear_merged_df.home_abbr.unique().tolist()
team_mapping = dict( zip(team_list,range(len(team_list))) )
dataYear_merged_df.replace({'away_abbr': team_mapping},inplace=True)
dataYear_merged_df.replace({'home_abbr': team_mapping},inplace=True)

save_merged_data = False
if save_merged_data:
    dataYear_merged_df.to_csv('Data/pre-processedData_scraped_inj_1920_n3.csv', index=False)

t4 = time.time()
t_features = (t4-t3)/60
print(t_features, 'minutes to merge injury data into historical features set')

t_tot = (t4-t1)/60
print(t_tot, 'minutes total to process data for', year_str)

"""
Ideas for who to quantify importance of each player and the effects of missing someone due to injury:
    Get the minutes played (divide by games played), usage percentage,
    offensive & defensive win shares (divide into units of per 48 minutes), and on-court 
    plus/minus (divide by games played) for each player.
    
    Calculate the fraction of each of these stats contributed by each player relative to the
    whole team.
    
    Given the injury report (ie which players are absent in a game), calculate the fraction
    of each of these stats which is missing from the team.
    
    Use these fractions as input features to the classifier model
    
"""
