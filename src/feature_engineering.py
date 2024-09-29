import pandas as pd
import numpy as np

def preprocessed_data(df):
    required_columns = ['player', 'season', 'pts', 'ast', 'trb', 'stl', 'blk', 'fg%', '3p%', 'ft%', 'mp', 'pos', 'tm']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Encode categorical variables
    for col in ['player', 'pos', 'tm']:
        df[col] = pd.factorize(df[col])[0]
    
      # Assuming season format is "2023-24"
    df.sort_values(by=['player', 'season'], inplace=True)

    # Year-over-Year Improvement
    improvement_columns = ['pts', 'ast', 'trb', 'stl', 'blk', 'fg%', '3p%', 'ft%']
    for col in improvement_columns:
        df[f'{col}_improvement'] = df.groupby('player')[col].pct_change()

    # Weighted Performance Averages
    for stat in ['pts', 'ast', 'trb']:
        df[f'weighted_{stat}'] = (
            df[stat] * (df['season'] == df['season'].max()) * 0.6 +
            df[stat] * (df['season'] == df['season'].max() - 1) * 0.3 +
            df[stat] * (df['season'] == df['season'].max() - 2) * 0.1
        )

    # Positional Rankings
    ranking_columns = ['pts', 'ast', 'trb', 'stl', 'blk', 'fg%', '3p%', 'ft%']
    for col in ranking_columns:
        df[f'{col}_positional_rank'] = df.groupby(['pos', 'season'])[col].rank(method='max', ascending=False)

    # Efficiency Metrics
    df['efficiency'] = (df['pts'] + df['ast'] + df['trb'] + df['stl'] + df['blk']) / df['mp']
    df['efficiency_rank'] = df.groupby('season')['efficiency'].rank(method='max', ascending=False)

    # Consistency Score
    df['consistency_score'] = df.groupby('player')['pts'].transform('std')
    df['consistency_rank'] = df.groupby('season')['consistency_score'].rank(method='max', ascending=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)  # Fill NaN values with 0
    
    
    # Separate the target (points) from the features
    y = df[['pts', 'ast', 'trb', 'stl', 'blk']]
    X = df.drop(columns=['pts', 'ast', 'trb', 'stl', 'blk'])


    return X, y


