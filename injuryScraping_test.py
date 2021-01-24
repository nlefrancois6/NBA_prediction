#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:07:00 2021

@author: noahlefrancois
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd

import NBApredFuncs as pf

#Df we want to fill in with scraped data
df = pd.DataFrame(columns=('Date','Team','Acquired','Relinquished','Notes'))

#url of the first page of search results
begin_date = '2020-12-10'
url = 'https://www.prosportstransactions.com/basketball/Search/SearchResults.php?Player=&Team=&BeginDate='+begin_date+'&EndDate=&ILChkBx=yes&Submit=Search'
#https://www.prosportstransactions.com/basketball/Search/SearchResults.php?Player=&Team=&BeginDate=2020-12-10&EndDate=&InjuriesChkBx=yes&Submit=Search
raw_data = requests.get(url)
soup = BeautifulSoup(raw_data.text, 'html.parser')

#get the list of results pages found in the tail of the first page
soup_a = soup.find_all('a')
for x in soup_a:
    try:
        #This will get overwritten each time so the end value will be the largest one
        numPages = int(x.string)
    except:
        dummy = 1

#Initialize storage arrays which will be used to fill df
d = []
t = []
a = []
r = []
n = []

#Scrape the first page and append it to our storage arrays
d, t, a, r, n = pf.injury_scrape_process_page(soup, d, t, a, r, n)

#Scrape the remaining pages and append them to the storage
for i in range(numPages-1):
    pageStart = 25*(i+1)

    soup_new_page = pf.injury_scrape_get_page_soup(pageStart, url)
    d, t, a, r, n = pf.injury_scrape_process_page(soup_new_page, d, t, a, r, n)  

#Load the scraped values into our df and save it to a csv
df['Date'] = d
df['Team'] = t
df['Acquired'] = a
df['Relinquished'] = r
df['Notes'] = n

df.to_csv('Data/injuries_2020_Jan2021.csv', index=False)
