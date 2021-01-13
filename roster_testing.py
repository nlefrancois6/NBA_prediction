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
from basketball_reference_scraper.injury_report import get_injury_report
    
t1 = time.time()
abbr_list = ['ATL', 'BOS', 'BRK','CHO','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']
#abbr_list = ['ATL', 'BRK']

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
    team_roster = Roster(team_abbr)
    player_list = team_roster.players
   
    i = 0
    games_team = 0
    minutes_team = 0
    usages_team = 0
    winShareO_team = 0
    winShareD_team = 0
    values_team = 0
    for player in team_roster.players:
        names.append(player.name)
        teams.append(team_abbr)
        #Note: if the player hasn't played yet this season, this will return the previous season which may not be a good representation
        #Need to check if the player has past stats
        career_df = player_list[i].dataframe
        if len(career_df) > 0:
            player_df = career_df.iloc[-2]
            #could use career_df.index.values[-2] to check whether the most recent season is this season or not
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
        
        i += 1

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
print(t_scrape, 'minutes to scrape')

#Get the season statistics of the first player on the roster, over the most recent season
#a_df = player_list[0].dataframe.iloc[-2]
#Do this in a loop w/ df.append in order to get a list of season statistics for the whole roster

injury_df = get_injury_report()

g_frac_team = []
m_frac_team = []
u_frac_team = []
wSO_frac_team = []
wSD_frac_team = []
v_frac_team = []
for team_abbr in abbr_list:
    #Get list of injured players on the team
    injuries_team = injury_df.loc[injury_df['TEAM'] == team_abbr]
    
    g_inj = 0
    m_inj = 0
    u_inj = 0
    wSO_inj = 0
    wSD_inj = 0
    v_inj = 0
    for i in range(len(injuries_team)):
        #Get the player name & stats from the injury report
        name = injuries_team['PLAYER'].iloc[i]
        #Need to check to make sure the player's name exists in league_players_df and handle if not. Simplest way is to not count them.
        inj_player = league_players_df.loc[league_players_df['name'] == name]
        if len(inj_player) > 0:
            g_inj += inj_player['games_played'].iloc[0]
            m_inj += inj_player['minutes_played'].iloc[0]
            u_inj += inj_player['usage_percentage'].iloc[0]
            wSO_inj += inj_player['offensive_win_shares'].iloc[0]
            wSD_inj += inj_player['defensive_win_shares'].iloc[0]
            v_inj += inj_player['value_over_replacement_player'].iloc[0]
            
            """
            print(name)
            print(g_inj)
            print(m_inj)
            print(u_inj)
            print(wSO_inj)
            print(wSD_inj)
            print(v_inj)
            """
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
    
    g_frac_team.append(g_frac)
    m_frac_team.append(m_frac)
    u_frac_team.append(u_frac)
    wSO_frac_team.append(wSO_frac)
    wSD_frac_team.append(wSD_frac)
    v_frac_team.append(v_frac)
    
d_inj = {'team': abbr_list,'games_played': g_frac_team,'minutes_played': m_frac_team, 'usage_percentage': u_frac_team, 'offensive_win_shares': wSO_frac_team, 'defensive_win_shares': wSD_frac_team, 'value_over_replacement_player': v_frac_team}
team_injLoss_df = pd.DataFrame(data=d_inj)

#save team_injLoss_df to a csv, use these columns as inputs to classifier
team_injLoss_df.to_csv('Data/injuryLosses2020.csv', index=False)

t3 = time.time()
t_process = (t3-t2)/60
print(t_process, 'minutes to process')
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
