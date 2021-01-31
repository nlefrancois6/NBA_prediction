#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:15:30 2020

@author: noahlefrancois
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sportsreference.nba.roster import Roster
from sportsreference.nba.boxscore import Boxscore
from basketball_reference_scraper.injury_report import get_injury_report
import math
#from keras import backend as K
#from keras.models import Model
#from keras.models import Sequential
#from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, Activation, concatenate
#from keras.optimizers import Adam

import requests
from bs4 import BeautifulSoup
import datetime
from datetime import date
import time
from pandas import DataFrame

def injury_scrape_process_page(soup, d, t, a, r, n):
    """
    Given the soup for a single page from prosportstransactions, scrape the 
    injury report data for every row.
    
    Return the data in each column as a separate array
    """
    all_entry_soup = soup.find('td').parent.find_next_siblings()

    numRows = len(soup.find_all(nowrap="nowrap"))
    for i in range(numRows):
        #Get a soup for all the columns in a given entry
        current_entry_soup = all_entry_soup[i]
        current_entry_values = current_entry_soup.find_all('td')
        
        #Extract each column value and save it to the respective list
        d.append(current_entry_values[0].string)
        t.append(current_entry_values[1].string[1:])
        a.append(current_entry_values[2].string[3:]) #Drop the \textbullet from the front
        r.append(current_entry_values[3].string[3:]) #Drop the \textbullet from the front
        n.append(current_entry_values[4].string)
        
    return d, t, a, r, n

def injury_scrape_get_page_soup(pageStart, url):
    """
    Get the soup from an injury page given the row to start at
    """
    if pageStart == 0:
        raw_data = requests.get(url)
        soup_page = BeautifulSoup(raw_data.text, 'html.parser')
    else:
        url_new_page = url+'&start='+str(pageStart)
        raw_data = requests.get(url_new_page)
        soup_page = BeautifulSoup(raw_data.text, 'html.parser')
    
    return soup_page

def reformat_scraped_odds(todays_odds, gmDate, verbose):
    """
    Take the odds for today obtained from the SBRscrape function and put them
    into the odds format we want (i.e. nba_odds_2021_scrape.csv)
    """

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
            if ((type(v_odds_row[s]) is np.float64) or (type(v_odds_row[s]) is np.int64)) and (math.isnan(v_odds_row[s]) is False) and (v_odds_row[s] < 10000000):
                v_odds.append(v_odds_row[s])
            if ((type(h_odds_row[s]) is np.float64) or (type(h_odds_row[s]) is np.int64)) and (math.isnan(v_odds_row[s]) is False) and (h_odds_row[s] < 10000000):
                h_odds.append(h_odds_row[s])
        
        """
        for s in range(len(v_odds_row)):
            if (type(v_odds_row[s]) is np.float64) and (math.isnan(v_odds_row[s]) is False):
                v_odds.append(v_odds_row[s])
            if (type(h_odds_row[s]) is np.float64) and (math.isnan(v_odds_row[s]) is False):
                h_odds.append(float(h_odds_row[s]))     
        """
        avg_v_odds = np.mean(v_odds)
        avg_h_odds = np.mean(h_odds)
        
        #Sometimes a team has both + and - odds for a game, resulting in abs(avg_odds)<100
        #If this occurs, replace the avg_odds with 100
        if np.abs(avg_v_odds) < 100:
            avg_v_odds = 100
        if np.abs(avg_h_odds) < 100:
            avg_h_odds = 100
        
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
    odds_df_oneRow = pd.DataFrame(data=d_odds)
    
    #Drop games with no odds (postponed, etc)
    if len(ind_nan) > 0:
        odds_df_oneRow = odds_df_oneRow.drop(ind_nan).reset_index().drop(columns = ['index'])
    
    #Relabel team names to abbreviations
    team_dict = {'Atlanta':'ATL', 'Boston':'BOS', 'Brooklyn':'BRK','Charlotte':'CHO','Chicago':'CHI','Cleveland':'CLE','Dallas':'DAL', 'Denver':'DEN','Detroit':'DET',
                 'Golden State':'GSW', 'Houston':'HOU', 'Indiana':'IND','L.A. Clippers':'LAC','L.A. Lakers':'LAL','Memphis':'MEM','Miami':'MIA','Milwaukee':'MIL','Minnesota':'MIN',
                 'New Orleans':'NOP','New York':'NYK','Oklahoma City':'OKC','Orlando':'ORL','Philadelphia':'PHI','Phoenix':'PHO','Portland':'POR','Sacramento':'SAC',
                 'San Antonio':'SAS','Toronto':'TOR','Utah':'UTA','Washington':'WAS'}
    
    for i in range(len(odds_df_oneRow)):
        odds_df_oneRow['away_abbr'][i] = team_dict[odds_df_oneRow['away_abbr'][i]]
        odds_df_oneRow['home_abbr'][i] = team_dict[odds_df_oneRow['home_abbr'][i]]
    
    #Put the odds into the two-row format
    odds = []
    team = []
    date = []
    vh = []
    final = []
    for i in range(len(odds_df_oneRow)):
        date.append(gmDate)
        date.append(gmDate)
        
        vh.append('V')
        vh.append('H')
        
        odds.append(odds_df_oneRow['V ML'][i])
        odds.append(odds_df_oneRow['H ML'][i])
        
        team.append(odds_df_oneRow['away_abbr'][i])
        team.append(odds_df_oneRow['home_abbr'][i])
        
        final.append(0)
        final.append(0)
       
    d_final = {'Date': date, 'VH': vh, 'Team': team, 'Final': final, 'ML': odds}  
    odds_df_twoRows = pd.DataFrame(data=d_final)
    
    return odds_df_oneRow, odds_df_twoRows

def get_stat_contributions(abbr_list, year_num, year_str):
    """
    Get the statistical contribution of every player on every team's roster during the specified season
    and calculate the total statistical contribution of each team's roster.
    
    Inputs:
        abbr_list: list containing abbreviation of each team
        year_num: season year as an int
        year_str: season year as a str

    Outputs: 
        league_players_df: df containing the list of player stat contributions
        team_totals_df: df containing the total statistical contributions of each team's roster
    """
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
    
    return league_players_df, team_totals_df

def scrape_todays_injury_report(today):
    """
    Scrape the injury report from today and return it in the correct format to be 
    used by get_injLoss_daily_report to get the missing production fraction
    """
    inj_report = get_injury_report()
    num_inj = len(inj_report)
    
    date = []
    name = []
    team = []
    for i in range(num_inj):
        date.append(today)
        name.append(inj_report['PLAYER'].iloc[i])
        team.append(inj_report['TEAM'].iloc[i])
          
    d_list = {'Date': date, 'Name': name, 'Team': team}
    inj_today = pd.DataFrame(data=d_list)
    
    return inj_today

def get_injLoss_daily_report(injYear_df, league_players_df, team_totals_df, abbr_list):
    """
    On each day in the season, get the list of injured players from each team and calculate 
    the fraction of the team's total statistics which are missing with the loss of those players
    
    Inputs:
        injYear_df: df containing the list of injured players for every day in the season
        league_players_df: df containing the list of player stat contributions
        team_totals_df: df containing the total statistical contributions of each team's roster
        abbr_list: list containing abbreviation of each team
    
    Output:
        team_injLoss_df: df containing, for every team on every day in the season, the 
        fraction of total team production which is missing due to injury
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
    
    return team_injLoss_df

def merge_injury_data(dataYear_df, team_injLoss_df):
    """
    For each game in dataYear_df, get the statistics from team_injLoss_df corresponding to both the
    home and away teams. Add these statistics as new columns to dataYear_df and save the resultant df
    
    Inputs:
        dataYear_df: df containing the team statistics for each game, which will be used as
        features in our model
        team_injLoss_df: df containing, for every team on every day in the season, the 
        fraction of total team production which is missing due to injury
        
    Output:
        dataYear_merged_df: df containing the team and injury statistics for each game, ready
        to be used to train and test our model
    """
    
    dataYear_merged_df = dataYear_df.copy()
    
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
    for i in range(len(dataYear_merged_df)):
        if i%100 == 0:
            print(i,'of',len(dataYear_merged_df),'games merged')
            
        away_abbr = dataYear_merged_df['away_abbr'][i]
        home_abbr = dataYear_merged_df['home_abbr'][i]
        gmDate = dataYear_merged_df['gmDate'][i]
        
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
        
    dataYear_merged_df['away_minutes_played_inj']  = m_a
    dataYear_merged_df['away_usage_percentage_inj']  = u_a
    dataYear_merged_df['away_offensive_win_shares_inj']  = wSO_a
    dataYear_merged_df['away_defensive_win_shares_inj']  = wSD_a
    dataYear_merged_df['away_value_over_replacement_inj']  = v_a
    
    dataYear_merged_df['home_minutes_played_inj']  = m_h
    dataYear_merged_df['home_usage_percentage_inj']  = u_h
    dataYear_merged_df['home_offensive_win_shares_inj']  = wSO_h
    dataYear_merged_df['home_defensive_win_shares_inj']  = wSD_h
    dataYear_merged_df['home_value_over_replacement_inj']  = v_h
    
    return dataYear_merged_df

def scrape_boxScore(schedule_dict, year, month, day, numDays):
    """
    Take the scraped game schedule from schedule_dict, scrape the corresponding
    boxscores, and combine them into a df
    """
    away_abbr = []
    home_abbr = []
    away_score = []
    home_score = []
    gmDate = []
    boxscore = []
    for i in range(numDays):
        date = str(month) + '-' + str(day) + '-' + str(year)
        print(date)
            
        #might need a try-except here to avoid crashing on a day with no games
        try:
            day_dict = schedule_dict[date]
            numGames_day = len(day_dict)
            
            for j in range(numGames_day):
                boxscore.append(day_dict[j]['boxscore'])
                away_abbr.append(day_dict[j]['away_abbr'])
                home_abbr.append(day_dict[j]['home_abbr'])
                away_score.append(day_dict[j]['away_score'])
                home_score.append(day_dict[j]['home_score'])
                
                if month > 9:
                    mstr = '-'
                else:
                    mstr = '-0'
                if day > 9:
                    dstr = '-'
                else:
                    dstr = '-0'
                gameDate = str(year) + mstr + str(month) + dstr + str(day)
                gmDate.append(gameDate)
                    
                if (j==0) and (i==0):
                    #for the first game on the first day,  initialize the df
                    game_df = Boxscore(day_dict[j]['boxscore']).dataframe
                    #Also save the initial date format
                else:
                    game_df_row = Boxscore(day_dict[j]['boxscore']).dataframe
                    game_df = pd.concat([game_df, game_df_row])
        except:
            print('No games on date', date)
                    
        #advance the date
        day, month, year = get_next_day(day, month, year)
        
            
    game_df['gmDate'] = gmDate
    game_df['away_abbr'] = away_abbr
    game_df['home_abbr'] = home_abbr
    game_df['away_score'] = away_score
    game_df['home_score'] = home_score
        
    return game_df

def preprocess_scraped_data(game_df, odds_df, inj_df, away_features, home_features, total_features, prev_num_games, encode_teams, year_int):
    """
    Take the boxscore and odds data including upcoming games, combine them, and perform
    processing such as averaging and inclusion of new columns
    Return the final game_df which will be saved and used in the model
    """
    
    
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
    v_odds_list, h_odds_list, v_score_list, h_score_list, broke_count = get_season_odds_matched_scrape(game_df, odds_df)
    #Add the odds as two new columns in stats_df
    game_df['V ML'] = v_odds_list
    game_df['H ML'] = h_odds_list
            
    
    #initialize storage for inj_df columns for each team
    hg = []
    hm = []
    hu = []
    hwSO = []
    hwSD = []
    hv = []
    
    vg = []
    vm = []
    vu = []
    vwSO = []
    vwSD = []
    vv = []
    for i in range(num_Games):
        #Get the teams and the date
        home_abbr = game_df['home_abbr'].iloc[i]
        away_abbr = game_df['away_abbr'].iloc[i]
        gmDate = game_df['gmDate'].iloc[i]
        
        #get inj_df rows for each team
        injDay_df = inj_df.loc[inj_df['date'] == gmDate]
        home_inj = injDay_df.loc[injDay_df['team'] == home_abbr]
        away_inj = injDay_df.loc[injDay_df['team'] == away_abbr]
        
        #store value of each column for each team
        hg.append(home_inj['games_played'].iloc[0])
        hm.append(home_inj['minutes_played'].iloc[0])
        hu.append(home_inj['usage_percentage'].iloc[0])
        hwSO.append(home_inj['offensive_win_shares'].iloc[0])
        hwSD.append(home_inj['defensive_win_shares'].iloc[0])
        hv.append(home_inj['value_over_replacement_player'].iloc[0])
        
        vg.append(away_inj['games_played'].iloc[0])
        vm.append(away_inj['minutes_played'].iloc[0])
        vu.append(away_inj['usage_percentage'].iloc[0])
        vwSO.append(away_inj['offensive_win_shares'].iloc[0])
        vwSD.append(away_inj['defensive_win_shares'].iloc[0])
        vv.append(away_inj['value_over_replacement_player'].iloc[0])
    
    #Merge the new columns into game_df
    #game_df['home_games_played_inj'] = hg
    game_df['home_minutes_played_inj'] = hm
    game_df['home_usage_percentage_inj'] = hu
    game_df['home_offensive_win_shares_inj'] = hwSO
    game_df['home_defensive_win_shares_inj'] = hwSD
    game_df['home_value_over_replacement_inj'] = hv
    
    #game_df['away_games_played_inj'] = vg
    game_df['away_minutes_played_inj'] = vm
    game_df['away_usage_percentage_inj'] = vu
    game_df['away_offensive_win_shares_inj'] = vwSO
    game_df['away_defensive_win_shares_inj'] = vwSD
    game_df['away_value_over_replacement_inj'] = vv
    
            
    #Get the rolling averages for our seasons of interest
    window = 'flat' #options are 'flat' or 'gaussian'
    avg = 'rolling' #options are 'rolling' or 'season'
    if avg == 'rolling':
        game_df = avg_previous_num_games(game_df, prev_num_games, window, home_features, away_features, team_list, scrape=True)
    if avg == 'season':
        game_df = avg_season(game_df, home_features, away_features, team_list, scrape=True)
            
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
            
    return game_df

def concat_upcoming_games(game_df, odds_df_oneRow, gmDate_boxscore):
    """
    Get the rows needed to add new games to game_df, corresponding to the new games in odds_df
    Return the updated game_df containing the new games
    Also return the df of just the new rows, for debugging
    """
    blankRow = pd.read_csv('Data/blank_row.csv')
    game_df_newRows = blankRow.copy()
    for i in range(len(odds_df_oneRow)):
        game_df_newRows['away_abbr'].iloc[i] = odds_df_oneRow['away_abbr'][i]
        game_df_newRows['home_abbr'].iloc[i] = odds_df_oneRow['home_abbr'][i]
        game_df_newRows['gmDate'].iloc[i] = gmDate_boxscore
        
        if i < (len(odds_df_oneRow)-1):
            game_df_newRows = pd.concat([game_df_newRows, blankRow])
    #Add the new rows into game_df, ready to be matched with the odds
    game_df = pd.concat([game_df, game_df_newRows], sort=False).reset_index().drop(columns = ['index'])
            
    return game_df, game_df_newRows
    
def get_next_day(day, month, year):
        
    months30 = [4,6,9,11]
    months31 = [1,3,5,7,8,10,12]
    leapyears = [2012, 2016, 2020]
    if month in months30:
        if day<30:
            day += 1
        else:
            month += 1
            day = 1
    elif month in months31:
        if day<31:
            day += 1
        else:
            if month == 12:
                year += 1
                month = 1
                day = 1
            else:
                month += 1
                day = 1           
    else:
        if year in leapyears:
            if day<29:
                day += 1
            else:
                month += 1
                day = 1
        else:
            if day<28:
                day += 1
            else:
                month += 1
                day = 1
    return day, month, year

def expected_gain(pred_prob, odds_v, odds_h):
    """
    Calculate the expected value of a bet
    
    pred_prob: Array of form [v_win_prob, v_lose_prob] with the model prediction probs
    
    odds_v: the moneyline odds on visitors
    
    odds_h: the moneyline odds on home
    """
    wager = 10
    
    #Get the predicted winner
    if pred_prob[0] > pred_prob[1]:
        visitor_win_pred = 'Win'
        correct_prob = pred_prob[0]
        wrong_prob = pred_prob[1]
    else:
        visitor_win_pred = 'Loss'
        correct_prob = pred_prob[1]
        wrong_prob = pred_prob[0]
    #Find the gain we would get if our prediction is correct
    if visitor_win_pred == 'Win':
        #odds[0] is odds on visitor win
        if odds_v > 0:
            gain_correct_pred = odds_v*(wager/100)
        else:
            gain_correct_pred = 100*(wager/(-odds_v))
    if pred_prob[0] < pred_prob[1]:
        #odds[1] is odds on home win
        if odds_h > 0:
            gain_correct_pred = odds_h*(wager/100)
        else:
            gain_correct_pred = 100*(wager/(-odds_h))
    #If our prediction is wrong, we lose the wager
    gain_wrong_pred = -wager
    #The expected gain is equal to each of the two possible gain outcomes multiplied
    #by the probability of the outcome, as determined by the model confidences
    exp_gain = gain_correct_pred*correct_prob + gain_wrong_pred*wrong_prob
    
    return exp_gain

def calc_Profit(account, wager_pct, fixed_wager, wager_crit, winner_prediction, winner_actual, moneyline_odds, regression_threshold, reg_profit_exp, expectation_threshold=False, scrape=False):
    """
    account: total money in the account at the start
    
    wager_pct: the amount wagered on each game as a fraction of the account. 
        float [0,1]
        
    fixed_wager: if True, just bet 10$ on every game. If False, use wager_pct to 
        calculate the amount to bet. In order to get meaningful/normalized labelling for
        the regression, need to set this to False.
    
    winner_prediction: the prediction of whether visiting team will win or lose.
        Possible values are 'Win' and 'Loss'
    
    winner_actual: the actual result of whether visiting team won or lost.
        Possible values are 'Win' and 'Loss' (might need to handle 'push')
    
    moneyline_odds: the moneyline odds given for visiting & home teams
        Not sure of format yet but probably (numGames,2) array with [V odds, H odds]
        Might need to apply a conversion for negative (ie favourite) odds, or handle the negative here
    
    pred_probs: the models confidence in each outcome. [prob V win, prob H win]
    
    expectation_threshold: If False, bet on every game. If a number, only bet when the 
        expected value of the bet is above that number.
    
    regression_threshold: If False, bet on every game. If a number, only bet when the 
        regression's expected value of the bet is above that number.
    
    reg_profit_exp: the regression's expected value of each bet
    
        
    Returns account_runningTotal, an array containing the total money we have after each game,
    and gains_store, a list of the gains from each bet.
    
    """
    #wins = 1
    #losses = 1
    first = True
    account_runningTotal = [account]
    gains_store = []
    gain = 0
    numGames = len(winner_prediction)
    for i in range(numGames):
        #By default, bet on the game (ie threshold_met = True)
        threshold_met = True
        #Check if an expected gain threshold is set
        if expectation_threshold != False:
            #Calculate the expected gain and check if it is above the set threshold
            #exp_gain = expected_gain(pred_probs[i], moneyline_odds[i,0], moneyline_odds[i,1])
            print('Expectation threshold is obsolete. Use regression threshold instead.')
            #if exp_gain < expectation_threshold:
                #If the threshold is not met, do not bet on the game
                #threshold_met = False
        #Check if a regression_threshold is set
        if regression_threshold != False:
            #Get the expected profit from the regression model and check if it's above the threshold
            exp_gain = reg_profit_exp[i]
            if exp_gain < regression_threshold:
                #If the threshold is not met, do not bet on the game
                threshold_met = False
        else:
            exp_gain = 5
        if threshold_met == True:
            if fixed_wager == False:
                #Seem to have best results with sqrt scaling
                #wager = wager_pct*100*np.exp(-exp_gain)
                if wager_crit == 'log':
                    wager = wager_pct*100*np.log(exp_gain/4+1.0001)
                if wager_crit == 'kelly':
                    #Could use these numbers as an initial value and continue to track & update them
                    #win_pct = wins/(wins+losses)
                    #w_l_ratio = avg_win/avg_loss
                    win_pct = 0.56
                    w_l_ratio = 1.28
                    k_pct = win_pct - (1-win_pct)/w_l_ratio #0.22
                    if first==True:
                        print(k_pct)
                        first = False
                    wager = k_pct*100 #could further multiply this by some f(exp_gain)
                    
                if wager_crit == 'sqrt':
                    wager = wager_pct*100*np.sqrt(exp_gain/5)
                
                if winner_prediction[i] == 'H':
                    if (moneyline_odds[i,1] < -120) and (wager > 4):
                        #print('rescale', wager)
                        scale_frac = -100/moneyline_odds[i,1]
                        wager = wager*scale_frac
                        #print('to', wager)
                elif winner_prediction[i] == 'V':
                    if (moneyline_odds[i,0] < -120) and (wager > 4):
                        #print('rescale', wager)
                        scale_frac = -100/moneyline_odds[i,0]
                        wager = wager*scale_frac
                        
                    
                if wager > 20:
                    wager = 20
            else:
                wager = 10
            #If our prediction was correct, calculate the winnings
            if winner_actual[i] == winner_prediction[i]:
                #wins += 1
                if scrape:
                    if winner_prediction[i] == 'V':
                        #odds[0] is odds on visitor win
                        if moneyline_odds[i,0]>0:
                            gain = moneyline_odds[i,0]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,0]))
                    if winner_prediction[i] == 'H':
                        #odds[1] is odds on home win
                        if moneyline_odds[i,1]>0:
                            gain = moneyline_odds[i,1]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,1]))
                else:
                    if winner_prediction[i] == 'Win':
                        #odds[0] is odds on visitor win
                        if moneyline_odds[i,0]>0:
                            gain = moneyline_odds[i,0]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,0]))
                    if winner_prediction[i] == 'Loss':
                        #odds[1] is odds on home win
                        if moneyline_odds[i,1]>0:
                            gain = moneyline_odds[i,1]*(wager/100)
                        else:
                            gain = 100*(wager/(-moneyline_odds[i,1]))
            #If our prediction was wrong, lose the wager
            else:
                #losses += 1
                gain = -wager
        
            account = account + gain
            account_runningTotal.append(account)
            gains_store.append(gain)
        
    return account_runningTotal, gains_store

def evaluate_model_profit(preds, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, fixed_wager, wager_crit, plot_gains, dummy_odds = False, regression_threshold=False, reg_profit_exp = False, scrape = False):
    """
    Take the predictions made by a model and a testing set with labels & the odds, calculate the
    final account balance and plot the account balance. Also return gains_store, a list of
    the gains from each bet.
    """
    
    account = 0
    winner_prediction = preds
    if scrape:
        winner_actual = testing_label['Winner'].array
    else:
        winner_actual = testing_label['teamRslt'].array
    
    #Get the ML odds for each game, either from the data or using dummy odds
    moneyline_odds = np.zeros([len(winner_prediction),2])
    if dummy_odds:
        for i in range(len(winner_prediction)):
            moneyline_odds[i,0] = 170
            moneyline_odds[i,1] = -200
    else:
        for i in range(len(winner_prediction)):
            moneyline_odds[i,0] = testing_odds_v.iloc[i]
            moneyline_odds[i,1] = testing_odds_h.iloc[i]

    account_runningTotal, gains_store = calc_Profit(account, wager_pct, fixed_wager, wager_crit, winner_prediction, winner_actual, moneyline_odds, regression_threshold, reg_profit_exp, expectation_threshold = min_exp_gain, scrape = scrape)
    
    #print('Final Account Balance after ', len(account_runningTotal), ' games: ', account_runningTotal[-1])
    
    if plot_gains:
        print('Final Account Balance after ', len(account_runningTotal), ' games: ', account_runningTotal[-1])
    
        plt.figure()
        plt.plot(account_runningTotal)
        plt.hlines(account, 0, len(account_runningTotal),linestyles='dashed')
        plt.xlabel('Games')
        plt.ylabel('Account Total')
        plt.title('Betting Performance of Our Model')
    
    return account_runningTotal, gains_store

def model_feature_importances(plot_features, model, features):
    """
    Plot and return the feature importances vector of a model
    """
    feature_importance = model.feature_importances_.tolist()
    if plot_features:
        plt.figure()
        plt.bar(features,feature_importance)
        plt.title('Feature Importance')
        plt.xticks(rotation='vertical')
        plt.show()
    
    return feature_importance

def avg_previous_num_games(df, num_games, window, home_features, away_features, team_list, scrape=False):
    # This function changes each stat to be the average of the last num_games for each team, and shifts it one so it does not include the current stats and drops the first num_games that become null
    if scrape:
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    else: 
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                if window == 'flat':
                    df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
                if window == 'gauss':
                    df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    return df.dropna()
"""
def avg_previous_num_games_scrape(df, num_games, window, home_features, away_features, team_list):
    # This function changes each stat to be the average of the last num_games for each team, and shifts it one so it does not include the current stats and drops the first num_games that become null
    for col in home_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            if window == 'flat':
                df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
            if window == 'gauss':
                df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    for col in away_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            if window == 'flat':
                df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3).mean()
            if window == 'gauss':
                df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).rolling(num_games, min_periods=3, win_type='gaussian').sum(std=1)/num_games
    return df.dropna()
"""
def avg_season(df, home_features, away_features, team_list, scrape=False):
    # This function changes each stat to be the season average for each team going into the game, and shifts it one so it does not include the current stats and drops the first num_games that become null
    if scrape:
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).expanding(min_periods=15).mean()
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).expanding(min_periods=15).mean()   
    else:
        for col in home_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['opptAbbr']==team] = df[col].loc[df['opptAbbr']==team].shift(1).expanding(min_periods=15).mean()
        for col in away_features:
            for team in team_list:
                #SettingWithCopyWarning raised but I don't think I care. Can take a look later
                df[col].loc[df['teamAbbr']==team] = df[col].loc[df['teamAbbr']==team].shift(1).expanding(min_periods=15).mean()
    return df.dropna()
"""
def avg_season_scrape(df, home_features, away_features, team_list):
    # This function changes each stat to be the season average for each team going into the game, and shifts it one so it does not include the current stats and drops the first num_games that become null
    for col in home_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            df[col].loc[df['home_abbr']==team] = df[col].loc[df['home_abbr']==team].shift(1).expanding(min_periods=15).mean()
    for col in away_features:
        for team in team_list:
            #SettingWithCopyWarning raised but I don't think I care. Can take a look later
            df[col].loc[df['away_abbr']==team] = df[col].loc[df['away_abbr']==team].shift(1).expanding(min_periods=15).mean()
    return df.dropna()
"""
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

def get_matching_game_score(stats_df_day, season, v_team, h_team, odds_df):
    """
    Take the date and the two teams for a game in stats_df, and get the rows of odds_df 
    containing the two teams from that game. Extract the score of each team from each row/team
    """
    sameday_odds_df = get_sameday_games(stats_df_day, season, odds_df)
    
    #Need to make sure 'Team' column is in same format
    v_odds_row = sameday_odds_df.loc[sameday_odds_df['Team'] == v_team]
    h_odds_row = sameday_odds_df.loc[sameday_odds_df['Team'] == h_team]
    
    v_score = v_odds_row['Final']
    h_score = h_odds_row['Final']
    
    return v_score, h_score


def get_season_odds_matched(stats_df_year, odds_df):
    """
    Given stats_df for a year, find the corresponding odds for every game and return
    the v and h odds in two arrays ready to be added to the df. Also return the number
    of games that encountered the size==0 error.
    """
    v_odds_list = []
    h_odds_list = []
    v_score_list = []
    h_score_list = []
    broke_count = 0
    for i in range(len(stats_df_year)):
        season = stats_df_year['Season'].iloc[i]
        game_date = stats_df_year['gmDate'].iloc[i]
        v_team = stats_df_year['teamAbbr'].iloc[i]
        h_team = stats_df_year['opptAbbr'].iloc[i]
        v_odds, h_odds = get_matching_game_odds(game_date, season, v_team, h_team, odds_df)
        v_score, h_score = get_matching_game_score(game_date, season, v_team, h_team, odds_df)
    
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
            
        v_score_num = v_score.iloc[0]
        h_score_num = h_score.iloc[0]
        
        v_odds_list.append(v_odds_num)
        h_odds_list.append(h_odds_num)
        v_score_list.append(v_score_num)
        h_score_list.append(h_score_num)
    
    return v_odds_list, h_odds_list, v_score_list, h_score_list, broke_count

def get_season_odds_matched_scrape(stats_df_year, odds_df):
    """
    Given stats_df for a year, find the corresponding odds for every game and return
    the v and h odds in two arrays ready to be added to the df. Also return the number
    of games that encountered the size==0 error.
    """
    v_odds_list = []
    h_odds_list = []
    v_score_list = []
    h_score_list = []
    broke_count = 0
    for i in range(len(stats_df_year)):
        noOdds = False
        season = stats_df_year['Season'].iloc[i]
        game_date = stats_df_year['gmDate'].iloc[i]
        v_team = stats_df_year['away_abbr'].iloc[i]
        h_team = stats_df_year['home_abbr'].iloc[i]
        v_odds, h_odds = get_matching_game_odds(game_date, season, v_team, h_team, odds_df)
        v_score, h_score = get_matching_game_score(game_date, season, v_team, h_team, odds_df)
        
        if (v_odds.size == 0) or (h_odds.size == 0):
            broke_count = broke_count + 1
            #Sometimes the odds have size 0. I think this is probably because the game isn't found
            #Maybe I'll want to drop that one, but for now just setting it to a neutral odds
            #should be fine
            v_odds_num = 0
            h_odds_num = 0
        else:
            v_odds_num = v_odds.iloc[0]
            h_odds_num = h_odds.iloc[0]
        
        if (v_score.size == 0) or (h_score.size == 0):
            noOdds = True
        
        if noOdds == False:
            v_score_num = v_score.iloc[0]
            h_score_num = h_score.iloc[0]
        else:
            v_score_num = 0
            h_score_num = 0
        
        v_odds_list.append(v_odds_num)
        h_odds_list.append(h_odds_num)
        v_score_list.append(v_score_num)
        h_score_list.append(h_score_num)
    
    return v_odds_list, h_odds_list, v_score_list, h_score_list, broke_count

def soup_url(type_of_line, tdate = str(date.today()).replace('-','')):
## get html code for odds based on desired line type and date
    if type_of_line == 'Spreads':
        url_addon = ''
    elif type_of_line == 'ML':
        url_addon = 'money-line/'
    elif type_of_line == 'Totals':
        url_addon = 'totals/'
    # elif type_of_line == '1H':
        # url_addon = '1st-half/'
    # elif type_of_line == '1HRL':
        # url_addon = 'pointspread/1st-half/'
    # elif type_of_line == '1Htotal':
        # url_addon = 'totals/1st-half/'
    else:
        print("Wrong url_addon")
    url = 'https://www.sportsbookreview.com/betting-odds/nba-basketball/' + url_addon + '?date=' + tdate
    #url = 'https://classic.sportsbookreview.com/betting-odds/nba-basketball/' + url_addon
    #now = datetime.datetime.now()
    raw_data = requests.get(url)
    soup_big = BeautifulSoup(raw_data.text, 'html.parser')
    soup = soup_big.find_all('div', id='OddsGridModule_5')[0]
    timestamp = time.strftime("%H:%M:%S")
    return soup, timestamp

def parse_and_write_data(soup, date, time, not_ML = True):
## Parse HTML to gather line data by book
    def book_line(book_id, line_id, homeaway):
        ## Get Line info from book ID
        line = soup.find_all('div', attrs = {'class':'el-div eventLine-book', 'rel':book_id})[line_id].find_all('div')[homeaway].get_text().strip()
        return line
    '''
    BookID  BookName
    238     Pinnacle
    19      5Dimes
    93      Bookmaker
    1096    BetOnline
    169     Heritage
    123     BetDSI
    999996  Bovada
    139     Youwager
    999991  SIA
    '''
    if not_ML:
        df = DataFrame(
                columns=('key','date','time',
                         'team','opp_team','pinnacle_line','pinnacle_odds',
                         '5dimes_line','5dimes_odds',
                         'heritage_line','heritage_odds',
                         'betonline_line','betonline_odds'))
    else:
        df = DataFrame(
            columns=('key','date','time',
                     'team',
                     'opp_team',
                     'pinnacle','5dimes','bookmaker',
                     'heritage','betonline','betDSI','youwager','SIA'))
    counter = 0
    number_of_games = len(soup.find_all('div', attrs = {'class':'el-div eventLine-rotation'}))
    print(number_of_games, 'games found')
    for i in range(0, number_of_games):
        A = []
        H = []
        #print('Game',str(i+1)+'/'+str(number_of_games))
        
        ## Gather all useful data from unique books
        # consensus_data = 	soup.find_all('div', 'el-div eventLine-consensus')[i].get_text()
        info_A = 		        soup.find_all('div', attrs = {'class':'el-div eventLine-team'})[i].find_all('div')[0].get_text().strip()
        # hyphen_A =              info_A.find('-')
        # paren_A =               info_A.find("(")
        team_A =                info_A
        # pitcher_A =             info_A[hyphen_A + 2 : paren_A - 1]
        # hand_A =                info_A[paren_A + 1 : -1]
        ## get line/odds info for unique book. Need error handling to account for blank data
        try:
            pinnacle_A = 	    book_line('238', i, 0)
        except IndexError:
            pinnacle_A = ''
        try:
            fivedimes_A = 	    book_line('19', i, 0)
        except IndexError:
            fivedimes_A = ''
        try:
            bookmaker_A =        book_line('93', i, 0)
        except IndexError:
            bookmaker_A = ''
        try:
            heritage_A =        book_line('169', i, 0)
        except IndexError:
            heritage_A = ''
        try:
            betonline_A = 		book_line('1096', i, 0)
        except IndexError:
            betonline_A = ''
        try:
            dsi_A = 		    book_line('123', i, 0)
        except IndexError:
            dsi_A = ''
        try:
            yw_A = 		    book_line('139', i, 0)
        except IndexError:
            yw_A = ''
        try:
            sia_A = 		    book_line('999991', i, 0)
        except IndexError:
            sia_A = ''
        info_H = 		        soup.find_all('div', attrs = {'class':'el-div eventLine-team'})[i].find_all('div')[1].get_text().strip()
        # hyphen_H =              info_H.find('-')
        # paren_H =               info_H.find("(")
        team_H =                info_H
        # pitcher_H =             info_H[hyphen_H + 2 : paren_H - 1]
        # hand_H =                info_H[paren_H + 1 : -1]
        try:
            pinnacle_H = 	    book_line('238', i, 1)
        except IndexError:
            pinnacle_H = ''
        try:
            fivedimes_H = 	    book_line('19', i, 1)
        except IndexError:
            fivedimes_H = ''
        try:
            bookmaker_H = 	    book_line('93', i, 1)
        except IndexError:
            bookmaker_H = '.'
        try:
            heritage_H = 	    book_line('169', i, 1)
        except IndexError:
            heritage_H = '.'
        try:
            betonline_H = 		book_line('1096', i, 1)
        except IndexError:
            betonline_H = ''
        try:
            dsi_H = 		    book_line('123', i, 1)
        except IndexError:
            dsi_H = ''
        try:
            yw_H = 		    book_line('139', i, 1)
        except IndexError:
            yw_H = ''
        try:
            sia_H = 		    book_line('999991', i, 1)
        except IndexError:
            sia_H = ''
        if team_H ==   'Detroit':
            team_H =   'Detroit'
        elif team_H == 'Indiana':
            team_H =   'Indiana'
        elif team_H == 'Brooklyn':
            team_H =   'Brooklyn'
        elif team_H == 'L.A. Lakers':
            team_H =   'L.A. Lakers'
        elif team_H == 'Washington':
            team_H =   'Washington'
        elif team_H == 'Miami':
            team_H =   'Miami'
        elif team_H == 'Minnesota':
            team_H =   'Minnesota'
        elif team_H == 'Chicago':
            team_H =   'Chicago'
        elif team_H == 'Oklahoma City':
            team_H =   'Oklahoma City'
        if team_A ==   'New Orleans':
            team_A =   'New Orleans'
        elif team_A == 'Houston':
            team_A =   'Houston'
        elif team_A == 'Dallas':
            team_A =   'Dallas'
        elif team_A == 'Cleveland':
            team_A =   'Cleveland'
        elif team_A == 'L.A. Clippers':
            team_A =   'L.A. Clippers'
        elif team_A == 'Golden State':
            team_A =   'Golden State'
        elif team_A == 'Denver':
            team_A =   'Denver'
        elif team_A == 'Boston':
            team_A =   'Boston'
        elif team_A == 'Milwaukee':
            team_A =   'Milwaukee'            
       # A.append(str(date) + '_' + team_A.replace(u'\xa0',' ') + '_' + team_H.replace(u'\xa0',' '))
        A.append(date)
        A.append(time)
        A.append('away')
        A.append(team_A)
        # A.append(pitcher_A)
        # A.append(hand_A)
        A.append(team_H)
        # A.append(pitcher_H)
        # A.append(hand_H)
        if not_ML:
            pinnacle_A = pinnacle_A.replace(u'\xa0',' ').replace(u'\xbd','.5')
            pinnacle_A_line = pinnacle_A[:pinnacle_A.find(' ')]
            pinnacle_A_odds = pinnacle_A[pinnacle_A.find(' ') + 1:]
            A.append(pinnacle_A_line)
            A.append(pinnacle_A_odds)
            fivedimes_A = fivedimes_A.replace(u'\xa0',' ').replace(u'\xbd','.5')
            fivedimes_A_line = fivedimes_A[:fivedimes_A.find(' ')]
            fivedimes_A_odds = fivedimes_A[fivedimes_A.find(' ') + 1:]
            A.append(fivedimes_A_line)
            A.append(fivedimes_A_odds)
            heritage_A = heritage_A.replace(u'\xa0',' ').replace(u'\xbd','.5')
            heritage_A_line = heritage_A[:heritage_A.find(' ')]
            heritage_A_odds = heritage_A[heritage_A.find(' ') + 1:]
            A.append(heritage_A_line)
            A.append(heritage_A_odds)
            betonline_A = betonline_A.replace(u'\xa0',' ').replace(u'\xbd','.5')
            betonline_A_line = betonline_A[:betonline_A.find(' ')]
            betonline_A_odds = betonline_A[betonline_A.find(' ') + 1:]
            A.append(betonline_A_line)
            A.append(betonline_A_odds)
        else:
            A.append(pinnacle_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(fivedimes_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(bookmaker_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(heritage_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(betonline_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(dsi_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(yw_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            A.append(sia_A.replace(u'\xa0',' ').replace(u'\xbd','.5'))
        #H.append(str(date) + '_' + team_A.replace(u'\xa0',' ') + '_' + team_H.replace(u'\xa0',' '))
        H.append(date)
        H.append(time)
        H.append('home')
        H.append(team_H)
        # H.append(pitcher_H)
        # H.append(hand_H)
        H.append(team_A)
        # H.append(pitcher_A)
        # H.append(hand_A)
        if not_ML:
            pinnacle_H = pinnacle_H.replace(u'\xa0',' ').replace(u'\xbd','.5')
            pinnacle_H_line = pinnacle_H[:pinnacle_H.find(' ')]
            pinnacle_H_odds = pinnacle_H[pinnacle_H.find(' ') + 1:]
            H.append(pinnacle_H_line)
            H.append(pinnacle_H_odds)
            fivedimes_H = fivedimes_H.replace(u'\xa0',' ').replace(u'\xbd','.5')
            fivedimes_H_line = fivedimes_H[:fivedimes_H.find(' ')]
            fivedimes_H_odds = fivedimes_H[fivedimes_H.find(' ') + 1:]
            H.append(fivedimes_H_line)
            H.append(fivedimes_H_odds)
            heritage_H = heritage_H.replace(u'\xa0',' ').replace(u'\xbd','.5')
            heritage_H_line = heritage_H[:heritage_H.find(' ')]
            heritage_H_odds = heritage_H[heritage_H.find(' ') + 1:]
            H.append(heritage_H_line)
            H.append(heritage_H_odds)
            betonline_H = betonline_H.replace(u'\xa0',' ').replace(u'\xbd','.5')
            betonline_H_line = betonline_H[:betonline_H.find(' ')]
            betonline_H_odds = betonline_H[betonline_H.find(' ') + 1:]
            H.append(betonline_H_line)
            H.append(betonline_H_odds)
        else:
            H.append(pinnacle_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(fivedimes_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(bookmaker_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(heritage_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(betonline_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(dsi_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(yw_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
            H.append(sia_H.replace(u'\xa0',' ').replace(u'\xbd','.5'))
        
	##For testing purposes..
	#for j in range(len(A)):
		#print 'Test: ', A[j]

        ## Take data from A and H (lists) and put them into DataFrame
        df.loc[counter]   = ([A[j] for j in range(len(A))])
        df.loc[counter+1] = ([H[j] for j in range(len(H))])
        counter += 2
    return df

def select_and_rename(df, text):
    ## Select only useful column names from a DataFrame
    ## Rename column names so that when merged, each df will be unique 
    if text[-2:] == 'ml':
        df = df[['key','time','team','opp_team',
                 'pinnacle','5dimes','bookmaker','heritage','betonline','dsi','youwager','sia']]
    ## Change column names to make them unique
        df.columns = ['key',text+'_time','team','opp_team',
                      text+'_PIN',text+'_FD',text+'_book',text+'_HER',text+'_BOL',text+'_dsi',text+'_yw',text+'_sia']
    else:
        df = df[['key','time','team','opp_team',
                 'pinnacle_line','pinnacle_odds',
                 '5dimes_line','5dimes_odds',
                 'heritage_line','heritage_odds',
                 'betonline_line','betonline_odds']]
        df.columns = ['key',text+'_time','team','opp_team',
                      text+'_PIN_line',text+'_PIN_odds',
                      text+'_FD_line',text+'_FD_odds',
                      text+'_HER_line',text+'_HER_odds',
                      text+'_BOL_line',text+'_BOL_odds']
    return df

def scrape_SBR_odds(filename):
    """
    Execute the functions from SBRscrape needed to get the ML odds from various
    books and save them to the specified file location as a csv.
    """
    # connectTor()

    ## Get today's lines
    todays_date = str(date.today()).replace('-','')
    ## change todays_date to be whatever date you want to pull in the format 'yyyymmdd'
    ## One could force user input and if results in blank, revert to today's date. 
    # todays_date = '20140611'

    ## store BeautifulSoup info for parsing
    soup_ml, time_ml = soup_url('ML', todays_date)
    print("Getting today's MoneyLine odds...")
    
    #### Each df_xx creates a data frame for a bet type
    #print("writing today's MoneyLine odds")
    df_ml = parse_and_write_data(soup_ml, todays_date, time_ml, not_ML = False)
    # print(df_ml)
    ## Change column names to make them unique
    df_ml.columns = ['key','date','ml_time','team',
                     'opp_team',
                     'ml_PIN','ml_FD','ml_book','ml_HER','ml_BOL','ml_dsi','ml_yw','ml_sia']  

    #Change this to change which columns are kept and remove books that aren't working
    filter_cols = ['key','date','ml_time','team',
                     'opp_team',
                     'ml_PIN','ml_FD','ml_book','ml_HER','ml_BOL']
    ## Merge all DataFrames together to allow for simple printout
    write_df = df_ml[filter_cols]
    
    write_df.to_csv(filename, index=False)#, header = False)
    
    print('Odds retrieval completed')

def gbcModel(training_features, training_label, testing_label, testing_features, n_est, learn_r, max_d):
    #Normalize the data
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)
    scaler.transform(testing_features)
    
    #Train a Gradient Boosting Machine on the data
    gbc = ensemble.GradientBoostingClassifier(n_estimators = n_est, learning_rate = learn_r, max_depth=max_d, subsample=1.0)
    gbc.fit(training_features, training_label)

    #Predict the outcome from our test set and evaluate the prediction accuracy for each model
    predGB = gbc.predict(testing_features) 
    pred_probsGB = gbc.predict_proba(testing_features) #probability of [results==True, results==False]
    accuracyGB = metrics.accuracy_score(testing_label, predGB)
    
    return gbc, predGB, pred_probsGB, accuracyGB

def rfcModel(training_features, training_label, testing_label, testing_features, n_est, rs, max_d):
    #Normalize the data
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)
    scaler.transform(testing_features)
    
    #Train a Random Forest on the data
    rfc = ensemble.RandomForestClassifier(n_estimators = n_est, max_depth=max_d, random_state = rs)
    rfc.fit(training_features, training_label)

    #Predict the outcome from our test set and evaluate the prediction accuracy for each model
    predRF = rfc.predict(testing_features) 
    pred_probsRF = rfc.predict_proba(testing_features) #probability of [results==True, results==False]
    accuracyRF = metrics.accuracy_score(testing_label, predRF)
    
    return rfc, predRF, pred_probsRF, accuracyRF

def decorrelation_loss(neuron, c):
    def loss(y_actual, y_predicted):
        return K.mean(
                K.square(y_actual-y_predicted) - c * K.square(y_predicted - neuron))
    return loss

def kerasModel(training_features, training_label, testing_label, testing_features, training_odds, testing_odds, c):
    encoder = LabelEncoder()
    encoder.fit(training_label)
    training_label_enc = encoder.transform(training_label)
    testing_label_enc = encoder.transform(testing_label)
    
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)
    scaler.transform(testing_features)
    
    """
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(60, activation='relu'),
      tf.keras.layers.Dense(30, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss = decorrelation_loss(training_odds, c),
                  #loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(training_features.values, training_label_enc, epochs=10)
    """
    model = Sequential()
    shape = training_features[0].shape
    print(shape)
    model.add(Conv2D(filters=32, kernel_size=(1, 8), input_shape=shape,
                     data_format="channels_first", activation="relu"))
    model.add(Flatten())
    
    model_input = Input(shape=shape)
    model_encoded = model(model_input)
    
    odds_input = Input(shape=(1,), dtype="float32") #(opening or closing weight)
    merged = concatenate([odds_input, model_encoded])
    output = Dense(32, activation="relu")(merged)
    output = Dropout(0.5)(output)
    output = Dense(8, activation="relu")(output)
    output = Dropout(0.5)(output)
    signal = Dense(1, activation="sigmoid")(output)
    
    opt = Adam(lr=0.0001)
    nba_model = Model(inputs=[model_input, odds_input], outputs=signal)
    print(nba_model.summary())
    
    nba_model.compile(optimizer=opt,
                      #loss="binary_crossentropy",
                  loss=decorrelation_loss(odds_input, c), # Call the loss function with the selected layer
                  metrics=['accuracy'])
    nba_model.fit([training_features, training_odds], training_label_enc,
                  batch_size=16,validation_data=([testing_features, testing_odds], testing_label_enc), verbose=1,epochs=20)
    accuracy = model.evaluate(testing_features, testing_label_enc)
    
    pred_probs = model.predict(testing_features)
    #Get the winner prediction from the model confidence. Not sure if this is done properly
    preds = []
    for i in range(len(pred_probs)):
        if pred_probs[i]>0.5:
            #not sure if it will always get encoded the same way. Should be a way to get the map from the encoder directly
            preds.append(['V'])
        else:
            preds.append(['H'])
    
    
    return model, preds, pred_probs, accuracy

def get_score_prediction(data_features, v_score_model, h_score_model):
    """
    Take two score prediction models and get their predictions on the games
    in data_features
    
    Outputs:    
        pred_winner: if 1, V win predicted. if 0, V loss predicted
        
        pred_margin: if positive, predicted margin of victory for V. if negative, 
            predicted margin of loss for V.
            
        v_score_pred, h_score_pred: predicted score for each team
            
    """
    v_score_pred = v_score_model.predict(data_features)
    h_score_pred = h_score_model.predict(data_features)
    #Get predicted winner and predicted margin on testing set
    numGames = len(v_score_pred)
    pred_winner = []
    for i in range(numGames):
        if v_score_pred[i] > h_score_pred[i]:
            pred_winner.append('Win')
        else:
            pred_winner.append('Loss')
    
    pred_margin = v_score_pred - h_score_pred
    
    return pred_winner, pred_margin, v_score_pred, h_score_pred

def score_prediction_model(training_features, training_label_v, training_label_h, testing_features, n_estimators, max_depth):
    """
    Fit two regression models to predict v points and h points, which can be used to
    predict spread, O/U, or ML
    """
    random_state = 1
    #Fit a regression model to predict visitor score
    v_score_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    v_score_model.fit(training_features, training_label_v)
    #Fit a regression model to predict home score
    h_score_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    h_score_model.fit(training_features, training_label_h)
    #Get predictions for score on testing set
    pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test = get_score_prediction(testing_features, v_score_model, h_score_model)
        
    return pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test, v_score_model, h_score_model

def layered_model_doubleReg_TrainTest(training_df, testing_df, class_features, output_label, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct):
    """
    Train and test the classification-regression layered model
    
    Inputs:
        training_df: Dataframe containing the training data
        
        testing_df: Dataframe containing the testing data
        
        class_features: array, features to use for the classifier model
        
        output_label: string, target output of the classifier model. Should be 'teamRslt'
        
        classifier_model: string, which classifier model to use
        
        class_params: array, hyperparameters for the classifier model
        
        reg_features: array, features to use for the regression model
        
        reg_label: string, target output of the regression model. Should be 'Classifier Profit'
        
        reg_params: array, hyperparameters for the regression model
        
        reg_threshold: float, minimum regression expectation required to place a bet
        
        plot_gains: boolean, if True plot the gains on the testing and validation set
            and print the final account balance w/ # of bets placed
            
        fixed_wager: boolean, if True just bet 10$ on every game
        
        wager_pct: float, if fixed_wager==False, bet balance*wager_pct
        
    Outputs:
        class_model: trained classification model object
            
        classifier_feature_importance: array, feature importance weightings for class_model
            
        profit_reg_model: trained regression model object
            
        testing_gains_reg: array, gain/loss from each game we bet upon
    """
    
    #Define the features (input) and label (prediction output) for training set
    training_features = training_df[class_features]
    #training_label = training_df[output_label]
    #training_odds_v = training_df['V ML'] #might not be needed
    #training_odds_h = training_df['H ML'] #might not be needed

    #Define features and label for testing set
    testing_features = testing_df[class_features]
    testing_label = testing_df[output_label]
    testing_odds_v = testing_df['V ML']
    testing_odds_h = testing_df['H ML']
    
    """
    This gives different outputs than the classifier, and will go into the profit regression
    differently. I think I should just write a whole new function to do this triple regression
    2-layer model, but that'll require me to change the name of this function and I don't
    want to mess with that right now.
    """
    #Train two regression models to predict V score and H score, use this to get
    #the predicted winner and predicted margin of victory
    training_label_v = training_df['V Score']
    training_label_h = training_df['H Score']
        
    n_estimators = 300
    max_depth = 5
        
    pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test, v_score_model, h_score_model = score_prediction_model(training_features, training_label_v, training_label_h, testing_features, n_estimators, max_depth)
     
    
    min_exp_gain = False
    plot_gains_score = False #This is hard-coded since I don't think I need it anymore
    running_account, testing_gains = evaluate_model_profit(pred_winner_test, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, fixed_wager, plot_gains_score, dummy_odds = False)
    if plot_gains_score:
        plt.title('Score Regression Model Profit, Testing')


    d = {'Classifier Profit': testing_gains, 'Pred Score V': v_score_pred_test, 'Pred Score H': h_score_pred_test, 'Prediction': pred_winner_test}
    reg_df = pd.DataFrame(data=d)

    reg_df['V ML'] = testing_odds_v.reset_index().drop(columns = ['index'])
    reg_df['H ML'] = testing_odds_h.reset_index().drop(columns = ['index'])
    
    #Separate the 2nd 1/2 of the data into training and testing data (I might end up using another year of data for testing instead)
    training_reg_df = reg_df.sample(frac=0.5, random_state=1)
    indlist_reg=list(training_reg_df.index.values)

    testing_reg_df = reg_df.copy().drop(index=indlist_reg)

    #Define the features and label for training set
    training_reg_features = training_reg_df[reg_features]
    training_reg_label = training_reg_df[reg_label]

    #Define features and label for testing set
    testing_reg_features = testing_reg_df[reg_features]
    #testing_reg_label = testing_reg_df[reg_label] #might not be needed
    
    #Create and train a regression model to predict the profit based on odds and classifier confidence
    #n_estimators = 100
    #max_depth = 3
    n_estimators = reg_params[0]
    max_depth = reg_params[1]
    
    random_state = 1
    
    profit_reg_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    #Train the model on the third 1/4 of the data
    profit_reg_model.fit(training_reg_features, training_reg_label)
    #Get the expected profit on the remaining 1/4 of the data
    expected_profit_testing = profit_reg_model.predict(testing_reg_features)

    #Do some dumb data formatting to get things in arrays even though alot of this doesn't get
    #used when we use the regression threshold instead of expected value
    preds_reg_testing = testing_reg_df['Prediction'].reset_index().drop(columns = ['index'])
    
    preds_reg_testing_arr = []
    for i in range(len(preds_reg_testing)):
        preds_reg_testing_arr.append(preds_reg_testing['Prediction'][i])
        
    testing_reg_odds_v = testing_reg_df['V ML'].reset_index().drop(columns = ['index'])
    testing_reg_odds_h = testing_reg_df['H ML'].reset_index().drop(columns = ['index'])

    #Evaluate the layered model profit on the remaining 1/4 of the testing data
    
    #Calculate the profit when we only bet on games the regression expectation favours 
    running_account_reg, testing_gains_reg = evaluate_model_profit(preds_reg_testing_arr, testing_label, testing_reg_odds_v, testing_reg_odds_h, min_exp_gain, wager_pct, fixed_wager, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_testing)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Testing Data')
    
    return v_score_model, h_score_model, profit_reg_model, testing_gains_reg


def layered_model_TrainTest(training_df, testing_df, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scrape):
    """
    Train and test the classification-regression layered model
    
    Inputs:
        training_df: Dataframe containing the training data
        
        testing_df: Dataframe containing the testing data
        
        class_features: array, features to use for the classifier model
        
        output_label: string, target output of the classifier model. Should be 'teamRslt'
        
        classifier_model: string, which classifier model to use
        
        class_params: array, hyperparameters for the classifier model
        
        reg_features: array, features to use for the regression model
        
        reg_label: string, target output of the regression model. Should be 'Classifier Profit'
        
        reg_params: array, hyperparameters for the regression model
        
        reg_threshold: float, minimum regression expectation required to place a bet
        
        plot_gains: boolean, if True plot the gains on the testing and validation set
            and print the final account balance w/ # of bets placed
            
        fixed_wager: boolean, if True just bet 10$ on every game
        
        wager_pct: float, if fixed_wager==False, bet balance*wager_pct
        
    Outputs:
        class_model: trained classification model object
            
        classifier_feature_importance: array, feature importance weightings for class_model
            
        profit_reg_model: trained regression model object
            
        testing_gains_reg: array, gain/loss from each game we bet upon
    """
    
    #Define the features (input) and label (prediction output) for training set
    training_features = training_df[class_features]
    training_label = training_df[output_label]
    #training_odds_v = training_df['V ML'] #might not be needed
    #training_odds_h = training_df['H ML'] #might not be needed

    #Define features and label for testing set
    testing_features = testing_df[class_features]
    testing_label = testing_df[output_label]
    testing_odds_v = testing_df['V ML']
    testing_odds_h = testing_df['H ML']
    
    if classifier_model == 'GB':
        #Train a Gradient Boosting Machine on the data, predict the outcomes, and evaluate accuracy
        #100, 0.02, 5 seems to work well. Also 500, 0.02, 5.
        #n_estimators = 500
        #learning_rate = 0.02
        #max_depth = 5
        n_estimators = class_params[0]
        learning_rate = class_params[1]
        max_depth = class_params[2]
        
        class_model, pred_class_test, pred_probs_class_test, accuracy_class_test = gbcModel(training_features, training_label, testing_label, testing_features, n_estimators, learning_rate, max_depth)
    elif classifier_model == 'RF':
        #Train a Random Forest Classifier on the data, predict the outcomes, and evaluate accuracy
        #n_estimators = 100
        #max_depth = 1
        n_estimators = class_params[0]
        max_depth = class_params[1]
        
        random_state = 1
        class_model, pred_class_test, pred_probs_class_test, accuracy_class_test = rfcModel(training_features, training_label, testing_label, testing_features, n_estimators, random_state, max_depth)
    elif classifier_model == 'Keras':
        #Probably need to handle the label encoding/decoding in other places
        class_model, pred_class_test, pred_probs_class_test, accuracy_class_test = kerasModel(training_features, training_label, testing_label, testing_features)
    """
    This gives different outputs than the classifier, and will go into the profit regression
    differently. I think I should just write a whole new function to do this triple regression
    2-layer model, but that'll require me to change the name of this function and I don't
    want to mess with that right now.
    elif classifier_model == 'Score Predict':
        
        training_label_v = training_df['V Score']
        training_label_h = training_df['H Score']
        
        n_estimators = 300
        max_depth = 5
        
        pred_winner_test, pred_margin_test, v_score_pred_test, h_score_pred_test, v_score_model, h_score_model = score_prediction_model(training_features, training_label_v, training_label_h, testing_features, n_estimators, max_depth)
     """   
        
    #Feature importance plots
    plot_features = False
    #classifier_feature_importance = model_feature_importances(plot_features, class_model, class_features)

    min_exp_gain = False
    
    plot_gains_class = False #This is hard-coded since I don't think I need it anymore
    fixed_wager_class = True
    running_account, testing_gains = evaluate_model_profit(pred_class_test, testing_label, testing_odds_v, testing_odds_h, min_exp_gain, wager_pct, fixed_wager_class, wager_crit, plot_gains_class, dummy_odds = False, scrape = scrape)
    if plot_gains_class:
        plt.title('Classifier Model Profit, Testing')

    if classifier_model == 'Keras':
        pred_probs_h = 1 - pred_probs_class_test
        d = {'Classifier Profit': testing_gains, 'Pred Probs V': pred_probs_class_test[:,0], 'Pred Probs H': pred_probs_h[:,0], 'Prediction': pred_class_test}
    else:
        #print(pred_probs_class_test.shape)
        d = {'Classifier Profit': testing_gains, 'Pred Probs V': pred_probs_class_test[:,0], 'Pred Probs H': pred_probs_class_test[:,1], 'Prediction': pred_class_test}
    reg_df = pd.DataFrame(data=d)
    #print(reg_df.shape)

    reg_df['V ML'] = testing_odds_v.reset_index().drop(columns = ['index'])
    reg_df['H ML'] = testing_odds_h.reset_index().drop(columns = ['index'])
    
    #Separate the 2nd 1/2 of the data into training and testing data (I might end up using another year of data for testing instead)
    training_reg_df = reg_df.sample(frac=0.5, random_state=1)
    indlist_reg=list(training_reg_df.index.values)

    testing_reg_df = reg_df.copy().drop(index=indlist_reg)

    #Define the features and label for training set
    training_reg_features = training_reg_df[reg_features]
    training_reg_label = training_reg_df[reg_label]

    #Define features and label for testing set
    testing_reg_features = testing_reg_df[reg_features]
    #testing_reg_label = testing_reg_df[reg_label] #might not be needed
    
    #Create and train a regression model to predict the profit based on odds and classifier confidence
    #n_estimators = 100
    #max_depth = 3
    n_estimators = reg_params[0]
    max_depth = reg_params[1]
    
    random_state = 1
    
    profit_reg_model = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, random_state=random_state)
    #Train the model on the third 1/4 of the data
    profit_reg_model.fit(training_reg_features, training_reg_label)
    #Get the expected profit on the remaining 1/4 of the data
    expected_profit_testing = profit_reg_model.predict(testing_reg_features)

    #Do some dumb data formatting to get things in arrays even though alot of this doesn't get
    #used when we use the regression threshold instead of expected value
    preds_reg_testing = testing_reg_df['Prediction'].reset_index().drop(columns = ['index'])
    pred_probs_reg_testing_v = testing_reg_df['Pred Probs V'].reset_index().drop(columns = ['index'])
    pred_probs_reg_testing_h = testing_reg_df['Pred Probs H'].reset_index().drop(columns = ['index'])

    preds_reg_testing_arr = []
    pred_probs_reg_testing = []
    for i in range(len(preds_reg_testing)):
        preds_reg_testing_arr.append(preds_reg_testing['Prediction'][i])
        pred_probs_reg_testing.append([pred_probs_reg_testing_v['Pred Probs V'][i], pred_probs_reg_testing_h['Pred Probs H'][i]])

    testing_reg_odds_v = testing_reg_df['V ML'].reset_index().drop(columns = ['index'])
    testing_reg_odds_h = testing_reg_df['H ML'].reset_index().drop(columns = ['index'])

    #Evaluate the layered model profit on the remaining 1/4 of the testing data
    
    #Calculate the profit when we only bet on games the regression expectation favours 
    running_account_reg, testing_gains_reg = evaluate_model_profit(preds_reg_testing_arr, testing_label, testing_reg_odds_v, testing_reg_odds_h, min_exp_gain, wager_pct, fixed_wager, wager_crit, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_testing, scrape = scrape)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Testing Data')
    
    return class_model, profit_reg_model, testing_gains_reg



def layered_model_validate(validation_data_df, class_features, output_label, class_model, reg_features, profit_reg_model, reg_threshold, plot_gains, fixed_wager, wager_pct, wager_crit, scrape):
    """
    Validate the layered model using an unseen dataset. Inputs are mostly the same as for 
    layered_model_TestTrain, with class_model and profit_reg_model the two trained model
    objects that make up our layered model.
    """
    
    #Format the validation data
    validation_class_features = validation_data_df[class_features]
    validation_label = validation_data_df[output_label]
    validation_odds_v = validation_data_df['V ML']
    validation_odds_h = validation_data_df['H ML']
    
    #Get classifier predictions on validation data & evaluate the gains
    pred_class_val = class_model.predict(validation_class_features) 
    pred_probs_class_val = class_model.predict_proba(validation_class_features)
    
    min_exp_gain = False
    plot_gains_class = False
    fixed_wager_class = True
    running_account_val, gains_val_class = evaluate_model_profit(pred_class_val, validation_label, validation_odds_v, validation_odds_h, min_exp_gain, wager_pct, fixed_wager_class, wager_crit, plot_gains_class, dummy_odds = False, scrape = scrape)
    if plot_gains_class:
        plt.title('Classifier Model Profit, Validation Data')
    
    #Data formatting for the regression
    d_val = {'Classifier Profit': gains_val_class, 'Pred Probs V': pred_probs_class_val[:,0], 'Pred Probs H': pred_probs_class_val[:,1], 'Prediction': pred_class_val}
    reg_val_df = pd.DataFrame(data=d_val)

    reg_val_df['V ML'] = validation_odds_v.reset_index().drop(columns = ['index'])
    reg_val_df['H ML'] = validation_odds_h.reset_index().drop(columns = ['index'])
    
    reg_val_features = reg_val_df[reg_features]
    
    #Get regression predictions on validation data
    expected_profit_val = profit_reg_model.predict(reg_val_features)
    
    #Data formatting for layered model profit evaluation
    preds_val_reg = reg_val_df['Prediction'].reset_index().drop(columns = ['index'])
    pred_probs_reg_val_v = reg_val_df['Pred Probs V'].reset_index().drop(columns = ['index'])
    pred_probs_reg_val_h = reg_val_df['Pred Probs H'].reset_index().drop(columns = ['index'])

    preds_val_reg_arr = []
    pred_probs_reg_val = []
    for i in range(len(preds_val_reg)):
        preds_val_reg_arr.append(preds_val_reg['Prediction'][i])
        pred_probs_reg_val.append([pred_probs_reg_val_v['Pred Probs V'][i], pred_probs_reg_val_h['Pred Probs H'][i]])

    val_reg_odds_v = reg_val_df['V ML'].reset_index().drop(columns = ['index'])
    val_reg_odds_h = reg_val_df['H ML'].reset_index().drop(columns = ['index'])
    
    running_account_reg_val, val_gains_reg = evaluate_model_profit(preds_val_reg_arr, validation_label, val_reg_odds_v, val_reg_odds_h, min_exp_gain, wager_pct, fixed_wager, wager_crit, plot_gains, dummy_odds = False, regression_threshold=reg_threshold, reg_profit_exp = expected_profit_val, scrape = scrape)
    if plot_gains:
        plt.title('Classification-Regression Layered Model Profit, Validation Data')
      
    return val_gains_reg

def make_new_bets(current_data_df, class_features, output_label, class_model, reg_features, reg_label, profit_reg_model, reg_threshold, fixed_wager, wager_pct, wager_crit, account):
    """
    Given data for some new games (current_data_df) and the trained layered model, 
    output the games that we should bet on and how much money to bet
    """
    
    #Format the data
    data_class_features = current_data_df[class_features]
    #data_label = current_data_df[output_label]
    data_odds_v = current_data_df['V ML']
    data_odds_h = current_data_df['H ML']
    
    pred_class = class_model.predict(data_class_features) 
    pred_probs_class = class_model.predict_proba(data_class_features)
    
    d_val = {'Pred Probs V': pred_probs_class[:,0], 'Pred Probs H': pred_probs_class[:,1], 'Prediction': pred_class}
    reg_data_df = pd.DataFrame(data=d_val)
    
    reg_data_df['V ML'] = data_odds_v.reset_index().drop(columns = ['index'])
    reg_data_df['H ML'] = data_odds_h.reset_index().drop(columns = ['index'])
    
    reg_data_features = reg_data_df[reg_features]
    expected_profit_data = profit_reg_model.predict(reg_data_features)

    bet_placed_index_store = []
    wager_store = []
    numGames = len(expected_profit_data)
    for i in range(numGames):
        #By default, bet on the game (ie threshold_met = True)
        threshold_met = True
        #Get the expected profit from the regression model and check if it's above the threshold
        exp_gain = expected_profit_data[i]
        if exp_gain < reg_threshold:
            #If the threshold is not met, do not bet on the game
            threshold_met = False
        if threshold_met == True:
            if fixed_wager == False:
                if wager_crit == 'log':
                    wager = wager_pct*account*np.log(exp_gain/4+1.0001)
                if wager_crit == 'kelly':
                    #Could use these numbers as an initial value and continue to track & update them
                    #win_pct = wins/(wins+losses)
                    #w_l_ratio = avg_win/avg_loss
                    win_pct = 0.56
                    w_l_ratio = 1.28
                    k_pct = win_pct - (1-win_pct)/w_l_ratio #0.22
                    wager = k_pct*account #could further multiply this by some f(exp_gain) 
                if wager_crit == 'sqrt':
                    wager = wager_pct*account*np.sqrt(exp_gain/5)
                    
                if wager > 20:
                    wager = 20
                if pred_class[i] == ['H']:
                    print('H')
            else:
                wager = 10
            
            bet_placed_index_store.append(i)
            wager_store.append(wager)
        else:
            wager_store.append(0)
            
    num_bets_placed = len(bet_placed_index_store)
    print(num_bets_placed, 'bets recommended out of', numGames, 'total games')
    
    return bet_placed_index_store, wager_store, pred_class
    
    
    