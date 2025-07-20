import pandas as pd
import numpy as np
from typing import List, Dict
from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def calculate_forward_std_devs_fast(df) -> pd.DataFrame:
    """Faster vectorized version using rolling windows."""
    
    horizons = {'3mo': 63, '6mo': 126, '9mo': 189, '12mo': 252, 
                '15mo': 315, '18mo': 378, '24mo': 504}
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    # Initialize all std columns
    for period in horizons:
        df[f'std{period}sfwds'] = np.nan
    
    # Process each ticker
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask, 'adj_close'].values
        n_obs = len(ticker_data)
        
        # Calculate for each horizon
        for period, window in horizons.items():
            std_values = np.full(n_obs, np.nan)
            
            for i in range(n_obs):
                end_idx = min(n_obs, i + window)
                if end_idx - i > 2:  # Need more than 2 observations
                    std_values[i] = np.std(ticker_data[i:end_idx], ddof=1)
            
            df.loc[mask, f'std{period}sfwds'] = std_values
    
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'