#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:03:24 2021

@author: noahlefrancois
"""

import pandas as pd
from datetime import datetime
from sportsreference.nba.boxscore import Boxscore, Boxscores

import NBApredFuncs as pf


games = Boxscores(datetime(2020, 12, 22), datetime(2021, 1, 3))
schedule_dict = games.games 
#each entry in the dict is another dict containing all the games from a single day
#Need to unpack this into a single dict before I can store it in a df
year = 2020
day = 22
month = 12
numDays = len(schedule_dict)

#Option to load data we already have from this year and just append the new data
update_year = False
if update_year:
    game_df = pd.read_csv('Data/scraped_boxScore_2020.csv')

away_abbr = []
home_abbr = []
away_score = []
home_score = []
gmDate = []
boxscore = []
#need to adapt it so this can loop through a whole year
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
            
            if (j==0) and (i==0) and (update_year == False):
                #for the first game on the first day, if we aren't loading data, initialize the df
                game_df = Boxscore(day_dict[j]['boxscore']).dataframe
            else:
                game_df_row = Boxscore(day_dict[j]['boxscore']).dataframe
                game_df = pd.concat([game_df, game_df_row])
    except:
        print('No games on date', date)
            
    #advance the date
    day, month, year = pf.get_next_day(day, month, year)

    
game_df['gmDate'] = gmDate
game_df['away_abbr'] = away_abbr
game_df['home_abbr'] = home_abbr
game_df['away_score'] = away_score
game_df['home_score'] = home_score


game_df.to_csv('Data/scraped_boxScore_2020.csv', index=False)
