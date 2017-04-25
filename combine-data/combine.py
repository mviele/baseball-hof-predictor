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
    df['AS'] = np.zeros((df_row_count,), dtype=np.int)

    df_hof = pd.read_csv('HallOfFame.csv', header=0)
    df_mem = df_hof.loc[df_hof['inducted'] == 'Y']

    for x in df_mem['playerID']:
        if x in df.index:
            df.ix[x, 'HOF'] = 1
    

    df_as = pd.read_csv('AllstarFull.csv', header=0)
    df_as = df_as.set_index('playerID')
    for p in df_as.index:
        if p in df.index:
            df.ix[p, 'AS'] += 1
            
    df_master = pd.read_csv('Master.csv', header=0)
    df_master = df_master.set_index('playerID')
    for p in df.index:
        if int(df_master.ix[p, 'finalGame'][:4]) <= 1938:
            df.ix[p, 'AS'] = np.nan

    print(df.loc[df['HOF'] == 1])
    df.to_csv("combined_stats.csv")



main()
