#Python 3

import pandas as pd


def main():
    df_init = pd.read_csv('Pitching.csv', header=0)

    df = df_init.drop(['yearID', 'stint', 'teamID', 'lgID', 'ERA', 'BAOpp'], axis=1, inplace=False)
    df = df.set_index('playerID')
    df = df.groupby(df.index).sum()
    df['ERA'] = (df['ER'] * 27) / df['IPouts']
    print(df)
    df.to_csv("combined_stats.csv")



main()
