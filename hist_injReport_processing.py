#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:10:14 2021

@author: noahlefrancois
"""

import pandas as pd
from basketball_reference_scraper.injury_report import get_injury_report
import NBApredFuncs as pf

read = False
if read:
    injury_df = get_injury_report()
    hist_df = pd.read_csv('Data/injuries_2010-2020.csv')

get_list = False
if get_list:
    dates_hist = hist_df.Date.unique()
    num_days = len(dates_hist)
    
    date = []
    team = []
    name = []
    in_out = []
    for day in dates_hist:
        #day = '2010-10-08'
        hist_day = hist_df.loc[hist_df['Date'] == day]
        num_inj = len(hist_day)
    
        for i in range(num_inj):
            t = hist_day['Team'].iloc[i]
            a = hist_day['Acquired'].iloc[i]
            if type(a) is str:
                n = hist_day['Acquired'].iloc[i]
                io = 'In'
            else:
                n = hist_day['Relinquished'].iloc[i]
                io = 'Out'
                  
            team.append(t)
            name.append(n)
            in_out.append(io)
            date.append(day)
    
    d_list = {'Date': date, 'Team': team, 'Name': name, 'In_Out': in_out}
    inj_list = pd.DataFrame(data=d_list)
    
    inj_list.to_csv('Data/injuries_historical_list.csv', index=False)

get_daily_report = True
if get_daily_report:
    inj_list = pd.read_csv('Data/injuries_historical_list.csv')
    
    date_d = []
    team_d = []
    name_d = []
    
    year = 2019
    day = 1
    month = 10
    
    if month > 9:
        mstr = '-'
    else:
        mstr = '-0'
    if day > 9:
        dstr = '-'
    else:
        dstr = '-0'
    gmDate = str(year) + mstr + str(month) + dstr + str(day)
    
    injured_name = []
    injured_team = []
    inj_day = inj_list.loc[inj_list['Date'] == gmDate]
    for i in range(len(inj_day)):
        i_o = inj_day['In_Out'].iloc[i]
        if i_o == 'Out':
            n = inj_day['Name'].iloc[i]
            t = inj_day['Team'].iloc[i]
            injured_name.append(n)
            injured_team.append(t)
    d_inj = {'Name': injured_name, 'Team': injured_team}
    inj_current = pd.DataFrame(data=d_inj)
    
    #advance the date
    day, month, year = pf.get_next_day(day, month, year)
    if month > 9:
        mstr = '-'
    else:
        mstr = '-0'
    if day > 9:
        dstr = '-'
    else:
        dstr = '-0'
    gmDate = str(year) + mstr + str(month) + dstr + str(day)
    
    print('Compiling daily injury report for each day in requested range...')
    #Not sure how to get the number of days, I guess I could do this for start date to end date
    #of each season
    numDays = 164
    #For each day
    for i in range(numDays):
        #print(gmDate)
        
        inj_day = inj_list.loc[inj_list['Date'] == gmDate]
        
        n_o = []
        t_o = []
        n_i = []
        t_i = []
        ind_i = []
        #Get the list of players in and players out on this day
        for j in range(len(inj_day)):
            i_o = inj_day['In_Out'].iloc[j]
            if i_o == 'Out':
                n_out = inj_day['Name'].iloc[j]
                t_out = inj_day['Team'].iloc[j]
                n_o.append(n_out)
                t_o.append(t_out)
            if i_o == 'In':
                n_in = inj_day['Name'].iloc[j]
                t_in = inj_day['Team'].iloc[j]
                row_in = inj_current.loc[inj_current['Name'] == n_in]
                if len(row_in) > 0:
                    ind = row_in.index[0]
                    
                    n_i.append(n_in)
                    t_i.append(t_in)
                    ind_i.append(ind)
        #Update the injury report by removing players 'in' and adding players 'out'
        #Add each player that is 'out'
        for k in range(len(n_o)):
            n_current = inj_current['Name'].array
            if n_o[k] not in n_current:
                inj_current = inj_current.append({'Name': n_o[k], 'Team': t_o[k]}, ignore_index=True)
        
        #Drop all of the 'in' players
        if len(ind_i) > 0:
            inj_current = inj_current.drop(ind_i).reset_index().drop(columns = ['index'])
        
        for l in range(len(inj_current)):
            n = inj_current['Name'].iloc[l]
            t = inj_current['Team'].iloc[l]
            
            date_d.append(gmDate)
            name_d.append(n)
            team_d.append(t)
        
        #advance the date
        day, month, year = pf.get_next_day(day, month, year)
        
        if month > 9:
            mstr = '-'
        else:
            mstr = '-0'
        if day > 9:
            dstr = '-'
        else:
            dstr = '-0'
        gmDate = str(year) + mstr + str(month) + dstr + str(day)
        

    print('List compiled.')
    d_inj_report = {'Date': date_d, 'Name': name_d, 'Team': team_d}
    inj_report_hist = pd.DataFrame(data=d_inj_report)
    
    inj_report_hist.to_csv('Data/injuries_historical_dailyReport_2019.csv', index=False)
    
    """
    Now I'll need to calculate the fraction of team production missing for each team on each day
    In order to do that, I'll need to use the method in roster_testing.py to get the season 
    statistics of every player from each year, extract the injury report on each day, and
    calculate the missing production for each team on that day.
    I'll need to load the preprocessed data containing all the games in our historical data sets,
    then for each game get the missing production of each team on that day. Then save this expanded
    data set back into a csv which can be used to train the model.
    """