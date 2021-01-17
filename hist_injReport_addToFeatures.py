#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:00:01 2021

@author: noahlefrancois
"""

import pandas as pd
import math
import time

from sportsreference.nba.roster import Roster
#from basketball_reference_scraper.injury_report import get_injury_report

year_num = 2019
year_str = '2019-20'

dataYear_df = pd.read_csv('Data/pre-processedData_scraped_abbr_1920_n3.csv')
injYear_df = pd.read_csv('Data/injuries_historical_dailyReport_2019.csv')

print('Relabelling Team Names...')
team_dict = {'Hawks':'ATL', 'Celtics':'BOS', 'Nets':'BRK','Hornets':'CHO','Bulls':'CHI','Cavaliers':'CLE','Mavericks':'DAL', 'Nuggets':'DEN','Pistons':'DET',
             'Warriors':'GSW', 'Rockets':'HOU', 'Pacers':'IND','Clippers':'LAC','Lakers':'LAL','Grizzlies':'MEM','Heat':'MIA','Bucks':'MIL','Timberwolves':'MIN',
             'Pelicans':'NOP','Knicks':'NYK','Thunder':'OKC','Magic':'ORL','76ers':'PHI','Suns':'PHO','Blazers':'POR','Kings':'SAC',
             'Spurs':'SAS','Raptors':'TOR','Jazz':'UTA','Wizards':'WAS'}

for i in range(len(injYear_df)):
    injYear_df['Team'][i] = team_dict[injYear_df['Team'][i]]

"""
Get the statistical contribution of every player on every team's roster during the specified season
and calculate the total statistical contribution of each team's roster.

Return league_players_df containing the list of player stat contributions, and team_totals_df
containing the total statistical contributions of each team's roster
"""

t1 = time.time()
abbr_list = ['ATL', 'BOS', 'BRK','CHO','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
#abbr_list = ['ATL', 'BOS']

names = []
teams = []
minutes = []
games = []
usages = []
winShareO = []
winShareD = []
values = []

games_teams = []
minutes_teams = []
usages_teams = []
winShareO_teams = []
winShareD_teams = []
values_teams = []
#This for-loop will likely be the slowest component, probably want to package this into a func and maybe put it in another script
for team_abbr in abbr_list: 
    print('Scraping statistical contributions for', team_abbr,'...') 
    team_roster = Roster(team_abbr, str(year_num))
    player_list = team_roster.players
   
    p = 0
    games_team = 0
    minutes_team = 0
    usages_team = 0
    winShareO_team = 0
    winShareD_team = 0
    values_team = 0
    for player in player_list:
        names.append(player.name)
        teams.append(team_abbr)
        #Note: if the player hasn't played yet this season, this will return the previous season which may not be a good representation
        #Need to check if the player has past stats
        career_df = player_list[p].dataframe
        if len(career_df) > 0:
            years_played = career_df.index.values
            year_found = False
            for i in range(len(years_played)):
                if years_played[i][0] == year_str:
                    year_found = True
                    year_ind = i
            if year_found:
                player_df = career_df.iloc[year_ind]
            
                g = player_df['games_played']
                if g is not None:
                    #Average the minutes and win shares by # games played
                    m = player_df['minutes_played']/g
                    #If m == 0 then u == None, so we need to change it to a zero
                    if m != 0:
                        u = player_df['usage_percentage']
                    else:
                        u = 0
                    wSO = player_df['offensive_win_shares']/g
                    wSD = player_df['offensive_win_shares']/g
                    #not sure if value needs to be averaged per game
                    v = player_df['value_over_replacement_player']
                else:
                    g = 0
                    m = 0
                    u = 0
                    wSO = 0
                    wSD = 0
                    v = 0
            else:
                g = 0
                m = 0
                u = 0
                wSO = 0
                wSD = 0
                v = 0
            
            if math.isnan(g):
                g = 0
                m = 0
                u = 0
                wSO = 0
                wSD = 0
                v = 0
            games.append(g)
            minutes.append(m)
            usages.append(u)
            winShareO.append(wSO)
            winShareD.append(wSD)
            values.append(v)

            #Track the total quantity for the team
            games_team += g
            minutes_team += m
            usages_team += u
            winShareO_team += wSO
            winShareD_team += wSD
            values_team += v
        if len(career_df) == 0:
            g = 0
            m = 0
            u = 0
            wSO = 0
            wSD = 0
            v = 0
            
            games.append(g)
            minutes.append(m)
            usages.append(u)
            winShareO.append(wSO)
            winShareD.append(wSD)
            values.append(v)
        
        p += 1

    games_teams.append(games_team)
    minutes_teams.append(minutes_team)
    usages_teams.append(usages_team)
    winShareO_teams.append(winShareO_team)
    winShareD_teams.append(winShareD_team)
    values_teams.append(values_team)


d_league = {'name': names, 'team': teams, 'games_played': games, 'minutes_played': minutes, 'usage_percentage': usages, 'offensive_win_shares': winShareO, 'defensive_win_shares': winShareD, 'value_over_replacement_player': values}  
league_players_df = pd.DataFrame(data=d_league)

d_teams = {'team': abbr_list,'games_played': games_teams,'minutes_played': minutes_teams, 'usage_percentage': usages_teams, 'offensive_win_shares': winShareO_teams, 'defensive_win_shares': winShareD_teams, 'value_over_replacement_player': values_teams}
team_totals_df = pd.DataFrame(data=d_teams)


t2 = time.time()
t_scrape = (t2-t1)/60
print(t_scrape, 'minutes to scrape. Scraping completed.')


"""
On each day in the season, get the list of injured players from each team and calculate 
the fraction of the team's total statistics which are missing with the loss of those players

Return a dataframe containing, for every team on every day in the season, the fraction of
total team production which is missing due to injury
"""

date_list = injYear_df.Date.unique()

team = []
date_team = []
g_frac_team = []
m_frac_team = []
u_frac_team = []
wSO_frac_team = []
wSD_frac_team = []
v_frac_team = []

print('Calculating injury losses for each day...')
date_list_test = date_list[0:1]
for date in date_list:
    injDay_df = injYear_df.loc[injYear_df['Date'] == date]
    #print(date)
    for team_abbr in abbr_list:
        #Get list of injured players on the team
        injuries_team = injDay_df.loc[injDay_df['Team'] == team_abbr]
        
        g_inj = 0
        m_inj = 0
        u_inj = 0
        wSO_inj = 0
        wSD_inj = 0
        v_inj = 0
        for i in range(len(injuries_team)):
            #Get the player name & stats from the injury report
            name = injuries_team['Name'].iloc[i]
            #Need to check to make sure the player's name exists in league_players_df and handle if not. Simplest way is to not count them.
            inj_player = league_players_df.loc[league_players_df['name'] == name]
            if len(inj_player) > 0:
                g_inj += inj_player['games_played'].iloc[0]
                m_inj += inj_player['minutes_played'].iloc[0]
                u_inj += inj_player['usage_percentage'].iloc[0]
                wSO_inj += inj_player['offensive_win_shares'].iloc[0]
                wSD_inj += inj_player['defensive_win_shares'].iloc[0]
                v_inj += inj_player['value_over_replacement_player'].iloc[0]
        
        #Calculate the fraction of team stats missing due to injury
        team_tots = team_totals_df.loc[team_totals_df['team'] == team_abbr]
        g_frac = g_inj/team_tots['games_played'].iloc[0]
        m_frac = m_inj/team_tots['minutes_played'].iloc[0]
        u_frac = u_inj/team_tots['usage_percentage'].iloc[0]
        wSO_frac = wSO_inj/team_tots['offensive_win_shares'].iloc[0]
        wSD_frac = wSD_inj/team_tots['defensive_win_shares'].iloc[0]
        v_frac = v_inj/team_tots['value_over_replacement_player'].iloc[0]
        
        if math.isnan(g_frac):
            g_frac = 0
            m_frac = 0
            u_frac = 0
            wSO_frac = 0
            wSD_frac = 0
            v_frac = 0
        
        team.append(team_abbr)
        date_team.append(date)
        g_frac_team.append(g_frac)
        m_frac_team.append(m_frac)
        u_frac_team.append(u_frac)
        wSO_frac_team.append(wSO_frac)
        wSD_frac_team.append(wSD_frac)
        v_frac_team.append(v_frac)

d_inj = {'date': date_team, 'team': team,'games_played': g_frac_team,'minutes_played': m_frac_team, 'usage_percentage': u_frac_team, 'offensive_win_shares': wSO_frac_team, 'defensive_win_shares': wSD_frac_team, 'value_over_replacement_player': v_frac_team}
team_injLoss_df = pd.DataFrame(data=d_inj)

#save team_injLoss_df to a csv, use these columns as inputs to classifier
team_injLoss_df.to_csv('Data/injuryLosses2019.csv', index=False)

t3 = time.time()
t_process = (t3-t2)/60
print(t_process, 'minutes to process daily injury report features')

#Probably want to combine team_injLoss_df with dataYear_df to get the full dataframe of features
#for each game in the year. Will probably need a loop to go through and get the injLoss for the
#two teams on the correct day for each game.

"""
For each game in dataYear_df, get the statistics from team_injLoss_df corresponding to both the
home and away teams. Add these statistics as new columns to dataYear_df and save the resultant df
"""
m_a = []
u_a = []
wSO_a = []
wSD_a = []
v_a = []
m_h = []
u_h = []
wSO_h = []
wSD_h = []
v_h = []
print('Merging injury data into historical features set...')
for i in range(len(dataYear_df)):
    if i%100 == 0:
        print(i,'of',len(dataYear_df),'games merged')
        
    away_abbr = dataYear_df['away_abbr'][i]
    home_abbr = dataYear_df['home_abbr'][i]
    gmDate = dataYear_df['gmDate'][i]
    
    injDay_df = team_injLoss_df.loc[team_injLoss_df['date'] == gmDate]
    away_row = injDay_df.loc[injDay_df['team'] == away_abbr]
    home_row = injDay_df.loc[injDay_df['team'] == home_abbr]
    
    m_a.append(away_row['minutes_played'].iloc[0])
    u_a.append(away_row['usage_percentage'].iloc[0])
    wSO_a.append(away_row['offensive_win_shares'].iloc[0])
    wSD_a.append(away_row['defensive_win_shares'].iloc[0])
    v_a.append(away_row['value_over_replacement_player'].iloc[0])
    
    m_h.append(home_row['minutes_played'].iloc[0])
    u_h.append(home_row['usage_percentage'].iloc[0])
    wSO_h.append(home_row['offensive_win_shares'].iloc[0])
    wSD_h.append(home_row['defensive_win_shares'].iloc[0])
    v_h.append(home_row['value_over_replacement_player'].iloc[0])
    
dataYear_df['away_minutes_played_inj']  = m_a
dataYear_df['away_usage_percentage_inj']  = u_a
dataYear_df['away_offensive_win_shares_inj']  = wSO_a
dataYear_df['away_defensive_win_shares_inj']  = wSD_a
dataYear_df['away_value_over_replacement_inj']  = v_a

dataYear_df['home_minutes_played_inj']  = m_h
dataYear_df['home_usage_percentage_inj']  = u_h
dataYear_df['home_offensive_win_shares_inj']  = wSO_h
dataYear_df['home_defensive_win_shares_inj']  = wSD_h
dataYear_df['home_value_over_replacement_inj']  = v_h


#Encode the team abbreviations back to numerical labels
team_list = dataYear_df.home_abbr.unique().tolist()
team_mapping = dict( zip(team_list,range(len(team_list))) )
dataYear_df.replace({'away_abbr': team_mapping},inplace=True)
dataYear_df.replace({'home_abbr': team_mapping},inplace=True)

dataYear_df.to_csv('Data/pre-processedData_scraped_inj_1920_n3.csv', index=False)

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
