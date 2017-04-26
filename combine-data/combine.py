#Python 3

import pandas as pd
import numpy as np


def main():
    df_init = pd.read_csv('Pitching.csv', header=0)

    df = df_init.drop(['yearID', 'stint', 'teamID', 'lgID', 'ERA', 'BAOpp', 'GIDP'], axis=1, inplace=False)
    df = df.set_index('playerID')
    df = df.groupby(df.index).sum()
    df = df[df['G'] >= 50]
    df['ERA'] = (df['ER'] * 27) / df['IPouts']


    df_row_count = df.shape[0]
    df['HOF'] = np.zeros((df_row_count,), dtype=np.int)

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df['LastYear'] = np.zeros((df_row_count,), dtype=np.int)
    df['AllStar'] = np.zeros((df_row_count,), dtype=np.int)
    df['MVP'] = np.zeros((df_row_count,), dtype=np.int)
    df['CyYoung'] = np.zeros((df_row_count,), dtype=np.int)
    df['WorldSeriesMVP'] = np.zeros((df_row_count,), dtype=np.int)
    df['GoldGlove'] = np.zeros((df_row_count,), dtype=np.int)

    df_master = pd.read_csv('Master.csv', header=0)
    df_master = df_master.set_index('playerID')

    df_hof = pd.read_csv('HallOfFame.csv', header=0)
    df_mem = df_hof.loc[df_hof['inducted'] == 'Y']

    for x in df_mem['playerID']:
        if x in df.index:
            df.loc[x, 'HOF'] = 1
    
    # All-Star game started in 1933
    df_as = pd.read_csv('AllstarFull.csv', header=0)
    df_as = df_as.set_index('playerID')
    for p in df_as.index:
        if p in df.index:
            df.loc[p, 'AllStar'] += 1
            

    df_awards = pd.read_csv('AwardsPlayers.csv', header=0)
    df_awards = df_awards.set_index('playerID')

    # MVP award started in 1911
    df_mvp = df_awards[df_awards['awardID'] == 'Most Valuable Player']
    for p in df_mvp.index:
        if p in df.index:
            df.loc[p, 'MVP'] += 1
    

    # Cy Young award started in 1956
    df_cy = df_awards[df_awards['awardID'] == 'Cy Young Award']
    for p in df_cy.index:
        if p in df.index:
            df.loc[p, 'CyYoung'] += 1
    
    
    # World Series MVP started in 1955
    df_wsmvp = df_awards[df_awards['awardID'] == 'World Series MVP']
    for p in df_wsmvp.index:
        if p in df.index:
            df.loc[p, 'WorldSeriesMVP'] += 1
    
    
    # Gold Glove started in 1957
    df_wsmvp = df_awards[df_awards['awardID'] == 'Gold Glove']
    for p in df_wsmvp.index:
        if p in df.index:
            df.loc[p, 'GoldGlove'] += 1
    
    
    for p in df.index:
        if pd.isnull(df_master.loc[p, 'finalGame']):
            df.loc[p, 'LastYear'] = 2017
        else:
            df.loc[p, 'LastYear'] = int(df_master.loc[p, 'finalGame'][:4])

    df_ineligable = df[df['LastYear'] >= 2011]
    df_eligable = df[df['LastYear'] < 2011]

    for p in df_eligable.index:
        if int(df_eligable.loc[p, 'LastYear']) <= 1961:
            df_eligable.loc[p, 'CyYoung'] = np.nan

    for p in df_eligable.index:
        if int(df_eligable.loc[p, 'LastYear']) <= 1960:
            df_eligable.loc[p, 'WorldSeriesMVP'] = np.nan

    for p in df_eligable.index:
        if int(df_eligable.loc[p, 'LastYear']) <= 1962:
            df_eligable.loc[p, 'GoldGlove'] = np.nan
    
    for p in df_eligable.index:
        if int(df_eligable.loc[p, 'LastYear']) <= 1916:
            df_eligable.loc[p, 'MVP'] = np.nan
    
    for p in df_eligable.index:
        if int(df_eligable.loc[p, 'LastYear']) <= 1938:
            df_eligable.loc[p, 'AllStar'] = np.nan

    df_ineligable.drop('HOF', axis=1, inplace=True)

    # We may want to leave this in, but for now we will drop this
    df_ineligable.drop('LastYear', axis=1, inplace=True)
    df_eligable.drop('LastYear', axis=1, inplace=True)

    print("Hall of Fame pitchers")
    print(df.loc[df['HOF'] == 1])
    df_eligable.to_csv("combined_stats_train.csv")
    df_eligable.to_csv("../predictor/combined_stats_train.csv")

    df_ineligable.to_csv("combined_stats_test.csv")
    df_ineligable.to_csv("../predictor/combined_stats_test.csv")



main()
