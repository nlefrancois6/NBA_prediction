#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:03:24 2021

@author: noahlefrancois
"""

import pandas as pd
from datetime import datetime
from sportsreference.nba.boxscore import Boxscore, Boxscores


games = Boxscores(datetime(2020, 12, 25), datetime(2021, 1, 2))
schedule_dict = games.games 
#each entry in the dict is another dict containing all the games from a single day
#Need to unpack this into a single dict before I can store it in a df
year = 2020
day = 25
month = 12
numDays = len(schedule_dict)

months30 = [4,6,9,11]
months31 = [1,3,5,7,8,10,12]
leapyears = [2012, 2016, 2020]


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
    
    day_dict = schedule_dict[date]
    numGames_day = len(day_dict)
    
    for j in range(numGames_day):
        boxscore.append(day_dict[j]['boxscore'])
        away_abbr.append(day_dict[j]['away_abbr'])
        home_abbr.append(day_dict[j]['home_abbr'])
        away_score.append(day_dict[j]['away_score'])
        home_score.append(day_dict[j]['home_score'])
        
        gameDate = str(year) + '-' + str(month) + '-' + str(day)
        gmDate.append(gameDate)
        
        if j==0 and i==0:
            game_df = Boxscore(day_dict[j]['boxscore']).dataframe
        else:
            game_df_row = Boxscore(day_dict[j]['boxscore']).dataframe
            game_df = pd.concat([game_df, game_df_row])
            
    #advance the date
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
            month += 1
            day = 1
            if month == 12:
                year += 1
                month = 1
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
    
    
game_df['away_abbr'] = away_abbr
game_df['home_abbr'] = home_abbr
game_df['away_score'] = away_score
game_df['home_score'] = home_score


game_df.to_csv('scraped_boxScore_2020.csv', index=False)
