#Python 3

import pandas as pd


def main():
    df_init = pd.read_csv('Pitching.csv', sep=',', header=0)
    
    df = df_init.loc[df_init['yearID'] == 1871]
    df.drop(['yearID', 'stint', 'teamID', 'lgID'], axis=1, inplace=False)

    for year in range(1872, 2016):
        df_new = df_init.loc[df_init['yearID'] == year]
        df_new.drop(['yearID', 'stint', 'teamID', 'lgID'], axis=1, inplace=False)
        df = pd.concat([df, df_new], axis=1)
    
    print(df)

main()
