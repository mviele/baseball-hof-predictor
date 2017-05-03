#Python 3

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('player_stats.csv', header=0)

    df.set_index('playerID', inplace=True)

    df_master = pd.read_csv('Master.csv', header=0)
    df_master = df_master.set_index('playerID')
    
    df_row_count = df.shape[0]
    df['Name'] = np.zeros((df_row_count,), dtype=np.int).astype(str)

    for p in df.index:
        df.loc[p, 'Name'] = df_master.loc[p, 'nameFirst'] + ' ' + df_master.loc[p, 'nameLast']
    
    df.to_csv('player_stats.csv')
        
def make_cols():
    df = pd.read_csv('player_stats.csv', header=0)

    df.set_index('playerID', inplace=True)
    
    df_row_count = df.shape[0]
    df['Mitchell-Report'] = np.zeros((df_row_count,), dtype=np.int).astype(str)
    df['Positive-Test'] = np.zeros((df_row_count,), dtype=np.int).astype(str)
    
    df.to_csv('player_stats_steroids.csv')



make_cols()
