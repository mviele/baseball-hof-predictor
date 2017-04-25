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
    print(df)

    df_row_count = df.shape[0]
    df['HOF'] = np.zeros((df_row_count,), dtype=np.int)
    print(df)
    df_hof = pd.read_csv('HallOfFame.csv', header=0)
    df_mem = df_hof.loc[df_hof['inducted'] == 'Y']
    print(df_mem)

    for x in df_mem['playerID']:
        if x in df.index:
            df.set_value(x, 'HOF', 1)
    
    print(df.loc[df['HOF'] == 1])
    # df.to_csv("combined_stats.csv")



main()
