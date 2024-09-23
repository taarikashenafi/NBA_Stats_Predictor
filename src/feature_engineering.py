import pandas as pd
import numpy as np

nba_df = pd.read_csv("./data/NBA_Regular_Season_Stats_2021-2024.csv")

#Year-over-Year Improvement
nba_df['season'] = nba_df['season'].str.slice(5, 9).astype(int)
nba_df.sort_values(by=['player', 'season'], inplace=True)
nba_df['pts_improvement'] = nba_df.groupby('player')['pts'].pct_change()
nba_df['ast_improvement'] = nba_df.groupby('player')['ast'].pct_change()
nba_df['trb_improvement'] = nba_df.groupby('player')['trb'].pct_change()
nba_df['stl_improvement'] = nba_df.groupby('player')['stl'].pct_change()
nba_df['blk_improvement'] = nba_df.groupby('player')['blk'].pct_change()
nba_df['fg%_improvement'] = nba_df.groupby('player')['fg%'].pct_change()
nba_df['3p%_improvement'] = nba_df.groupby('player')['3p%'].pct_change()
nba_df['ft%_improvement'] = nba_df.groupby('player')['ft%'].pct_change()

# Weighted Performance Averages
nba_df['weighted_pts'] = nba_df['pts'] * (nba_df['season'] == 2024) * 0.6 + \
                         nba_df['pts'] * (nba_df['season'] == 2023) * 0.3 + \
                         nba_df['pts'] * (nba_df['season'] == 2022) * 0.1

nba_df['weighted_ast'] = nba_df['ast'] * (nba_df['season'] == 2024) * 0.6 + \
                         nba_df['ast'] * (nba_df['season'] == 2023) * 0.3 + \
                         nba_df['ast'] * (nba_df['season'] == 2022) * 0.1

nba_df['weighted_trb'] = nba_df['trb'] * (nba_df['season'] == 2024) * 0.6 + \
                         nba_df['trb'] * (nba_df['season'] == 2023) * 0.3 + \
                         nba_df['trb'] * (nba_df['season'] == 2022) * 0.1

#Positional Rankings
nba_df['pts_positional_rank'] = nba_df.groupby(['pos', 'season'])['pts'].rank(method='max', ascending=False)
nba_df['ast_positional_rank'] = nba_df.groupby(['pos', 'season'])['ast'].rank(method='max', ascending=False)
nba_df['trb_positional_rank'] = nba_df.groupby(['pos', 'season'])['trb'].rank(method='max', ascending=False)
nba_df['stl_positional_rank'] = nba_df.groupby(['pos', 'season'])['stl'].rank(method='max', ascending=False)
nba_df['blk_positional_rank'] = nba_df.groupby(['pos', 'season'])['blk'].rank(method='max', ascending=False)
nba_df['fg%_positional_rank'] = nba_df.groupby(['pos', 'season'])['fg%'].rank(method='max', ascending=False)
nba_df['3p%_positional_rank'] = nba_df.groupby(['pos', 'season'])['3p%'].rank(method='max', ascending=False)
nba_df['ft%_positional_rank'] = nba_df.groupby(['pos', 'season'])['ft%'].rank(method='max', ascending=False)

# Efficiency Metrics
nba_df['efficiency'] = (nba_df['pts'] + nba_df['ast'] + nba_df['trb'] + nba_df['stl'] + nba_df['blk']) / nba_df['mp']
nba_df['efficiency_rank'] = nba_df.groupby('season')['efficiency'].rank(method='max', ascending=False)

#Consistency Score
nba_df['consistency_score'] = nba_df.groupby('player')['pts'].transform('std')
nba_df['consistency_rank'] = nba_df.groupby('season')['consistency_score'].rank(method='max', ascending=False)

nba_df.replace([np.inf, -np.inf], np.nan, inplace=True)

nba_df.to_csv('./data/NBA_Feature_Engineered_Data.csv', index=False)