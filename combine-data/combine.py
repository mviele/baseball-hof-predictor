#Python 3

import pandas as pd
import numpy as np


def main():
    df_init = pd.read_csv('Pitching.csv', header=0)

    df = df_init.drop(['yearID', 'stint', 'teamID', 'lgID', 'ERA', 'BAOpp'], axis=1, inplace=False)
    df = df.set_index('playerID')
    df = df.groupby(df.index).sum()
    df = df[df['G'] >= 50]
    df['ERA'] = (df['ER'] * 27) / df['IPouts']


    df_row_count = df.shape[0]
    df['HOF'] = np.zeros((df_row_count,), dtype=np.int)
    df['AllStar'] = np.zeros((df_row_count,), dtype=np.int)
    df['MVP'] = np.zeros((df_row_count,), dtype=np.int)
    df['CyYoung'] = np.zeros((df_row_count,), dtype=np.int)
    df['WorldSeriesMVP'] = np.zeros((df_row_count,), dtype=np.int)
    df['GoldGlove'] = np.zeros((df_row_count,), dtype=np.int)

    df_hof = pd.read_csv('HallOfFame.csv', header=0)
    df_mem = df_hof.loc[df_hof['inducted'] == 'Y']

    for x in df_mem['playerID']:
        if x in df.index:
            df.ix[x, 'HOF'] = 1
    
    # All-Star game started in 1933
    df_as = pd.read_csv('AllstarFull.csv', header=0)
    df_as = df_as.set_index('playerID')
    for p in df_as.index:
        if p in df.index:
            df.ix[p, 'AllStar'] += 1
            
    df_master = pd.read_csv('Master.csv', header=0)
    df_master = df_master.set_index('playerID')
    for p in df.index:
        if int(df_master.ix[p, 'finalGame'][:4]) <= 1938:
            df.ix[p, 'AllStar'] = np.nan

    df_awards = pd.read_csv('AwardsPlayers.csv', header=0)
    df_awards = df_awards.set_index('playerID')

    # MVP award started in 1911
    df_mvp = df_awards[df_awards['awardID'] == 'Most Valuable Player']
    for p in df_mvp.index:
        if p in df.index:
            df.ix[p, 'MVP'] += 1
    
    for p in df.index:
        if int(df_master.ix[p, 'finalGame'][:4]) <= 1916:
            df.ix[p, 'MVP'] = np.nan

    # Cy Young award started in 1956
    df_cy = df_awards[df_awards['awardID'] == 'Cy Young Award']
    for p in df_cy.index:
        if p in df.index:
            df.ix[p, 'CyYoung'] += 1
    
    for p in df.index:
        if int(df_master.ix[p, 'finalGame'][:4]) <= 1961:
            df.ix[p, 'CyYoung'] = np.nan

    # World Series MVP started in 1955
    df_wsmvp = df_awards[df_awards['awardID'] == 'World Series MVP']
    for p in df_wsmvp.index:
        if p in df.index:
            df.ix[p, 'WorldSeriesMVP'] += 1
    
    for p in df.index:
        if int(df_master.ix[p, 'finalGame'][:4]) <= 1960:
            df.ix[p, 'WorldSeriesMVP'] = np.nan

    # Gold Glove started in 1957
    df_wsmvp = df_awards[df_awards['awardID'] == 'Gold Glove']
    for p in df_wsmvp.index:
        if p in df.index:
            df.ix[p, 'GoldGlove'] += 1
    
    for p in df.index:
        if int(df_master.ix[p, 'finalGame'][:4]) <= 1962:
            df.ix[p, 'GoldGlove'] = np.nan


    print(df.loc[df['HOF'] == 1])
    df.to_csv("combined_stats.csv")



main()
