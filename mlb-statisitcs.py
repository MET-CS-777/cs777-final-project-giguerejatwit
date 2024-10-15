"""
author: Jake Giguere <giguere@bu.edu>
Get player batting data from 2014-2024 and create a new csv file
"""
from io import StringIO
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_player_batting_data(year: int) -> pd.DataFrame:
    url = f"https://www.baseball-reference.com/leagues/majors/{year}-standard-batting.shtml"
    try:
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:
        print(f'Could not get player data from year {year}')

    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table', {'id': 'players_standard_batting'})
    table = StringIO(str(table))
    
    df = pd.read_html(table)[0]

    return df


def get_all_data(start_year:int=2014, end_year:int = 2024) -> pd.DataFrame:
    print("Fetching data 🦴...")
    """
    Concatenates all of the dataframes and returns a dataframe of all year within the parameters

    Args:
        start_year: integer of the earliest year to start the loop from
        end_year: integer of the lastest year to end the loop at
    """
    df = pd.DataFrame()
    for year in range(start_year, end_year):
        year_df = get_player_batting_data(year)
        df = pd.concat(objs=[df, year_df], axis=0, ignore_index=True)

    clean_df = clean_data(df)
    return clean_df

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    print("cleaning 🧼 ...  ")
    """
    Cleans NaN values, rows with 0's, and unessecary columns
    Unessecary columns: [Rk,Age,Team,Pos,Awards]
    Unessecary rows: [League Average]
    """

    #Drop rows with No Batting average
    data = data[data['G'] > 10]
    data = data[data['BA'] != 0]

    #check if there are any NaN values
    print(f'Null Values? {data.isnull().values.any()}')
    print(f'NaN Values? {data.isna().values.any()}')

    #Drop unnessecary columns
    ex_cols = ["Rk","Age","Team","Pos","Awards"]
    data.drop(columns=ex_cols, axis=1, inplace=True)

    
    #Drop rows that start with "League Average" in a specific column
    data = data[~data['Player'].str.startswith('League Average')]

    #reset the index after dropping the rows
    data.reset_index(drop=True, inplace=True)

    return data



if __name__ == '__main__':
    #check if file exists else run the functions
    filepath = 'Data/batting_2014_2024.csv'
    if os.path.exists(filepath):
        print("Found some data 🕵️‍♂️ ...")
        df = pd.read_csv(filepath, delimiter=',', header=0)
        clean_data(df)

    else:
        print("File not found 🚫")
        df = get_all_data(2014, 2024)
        df.to_csv('Data/batting_2014_2024.csv')
        print(f'Data saved to {filepath} ✅')
