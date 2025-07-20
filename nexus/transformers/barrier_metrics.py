import pandas as pd
import numpy as np
from typing import List, Dict

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def calculate_barrier_metrics(df, *args, **kwargs
) -> pd.DataFrame:
    """
    Calculate comprehensive barrier breach metrics for panel stock data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel data with columns: ['date', 'ticker', 'adjusted_close']
        Must be sorted by ticker, then date
    barriers : List[float]
        List of barrier levels as fractions (e.g., [0.85, 0.80, 0.75, 0.70])
    trading_days_per_year : int, default 252
        Number of trading days per year for timeframe conversion
    timeframes_months : List[int], default [3, 6, 9, 12, 15, 18, 24]
        Timeframes in months to calculate
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with additional columns:
        - mean_bb_price{timeframe}mos{barrier_num}: Mean price during barrier breaches
        - pct_above{timeframe}mos{barrier_num}: Percentage of days above barrier
        - pct_below{timeframe}mos{barrier_num}: Percentage of days below barrier
    """
    
    # Validate inputs
    
    required_cols = ['date', 'ticker', 'adj_close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    barriers = kwargs['configuration'].get('barriers')

    if not barriers:
        raise ValueError("At least one barrier level must be provided")
    
    timeframes_months = kwargs['configuration'].get('timeframes_months')

    trading_days_per_year = kwargs['configuration'].get('trading_days_per_year')
    # Convert timeframes to trading days
    timeframes_days = {
        months: int(months * trading_days_per_year / 12) 
        for months in timeframes_months
    }
    
    # Ensure data is sorted
    df_sorted = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Initialize result columns
    result_cols = {}
    for months in timeframes_months:
        for i, barrier in enumerate(barriers, 1):
            result_cols[f'mean_bb_price{months}mos{i}'] = [np.nan] * len(df_sorted)
            result_cols[f'pct_above{months}mos{i}'] = [np.nan] * len(df_sorted)
            result_cols[f'pct_below{months}mos{i}'] = [np.nan] * len(df_sorted)
    
    # Group by ticker for panel processing
    grouped = df_sorted.groupby('ticker')
    
    for ticker, group in grouped:
        print(f"Processing {ticker}...")
        
        # Convert to numpy arrays for faster computation
        prices = group['adj_close'].values
        ticker_indices = group.index.values
        n_obs = len(prices)
        
        # For each observation date
        for i in range(n_obs):
            current_price = prices[i]
            
            # Calculate barrier prices for this observation
            barrier_prices = [current_price * barrier for barrier in barriers]
            
            # For each timeframe
            for months in timeframes_months:
                days_ahead = timeframes_days[months]
                
                # Define forward-looking window - SAS style: start from i (inclusive)
                # toploop = min(observations, target_days) but from current position
                remaining_obs = n_obs - i
                actual_window_size = min(remaining_obs, days_ahead)
                
                # Extract forward-looking prices INCLUDING current observation (i)
                # This matches SAS: do j=i to toploop where j starts at i
                forward_prices = prices[i:i + actual_window_size]
                total_days = len(forward_prices)
                
                # Skip if no forward data
                if total_days == 0:
                    continue
                
                original_idx = ticker_indices[i]
                
                # For each barrier level
                for barrier_idx, barrier_price in enumerate(barrier_prices, 1):
                    
                    # Calculate above/below barrier masks
                    above_barrier_mask = forward_prices >= forward_prices[0]
                    below_barrier_mask = forward_prices < barrier_price
                    
                    # Calculate percentages
                    days_above = np.sum(above_barrier_mask)
                    days_below = np.sum(below_barrier_mask)
                    
                    pct_above = days_above / total_days if total_days > 0 else 0
                    pct_below = days_below / total_days if total_days > 0 else 0
                    
                    # Store percentage results
                    result_cols[f'pct_above{months}mos{barrier_idx}'][original_idx] = pct_above
                    result_cols[f'pct_below{months}mos{barrier_idx}'][original_idx] = pct_below
                    
                    # Calculate mean breach price (only if there were breaches)
                    if days_below > 0:
                        below_barrier_prices = forward_prices[below_barrier_mask]
                        mean_bb_price = np.mean(below_barrier_prices)
                        result_cols[f'mean_bb_price{months}mos{barrier_idx}'][original_idx] = mean_bb_price
    
    # Add calculated columns to original dataframe
    result_df = df_sorted.copy()
    for col_name, values in result_cols.items():
        result_df[col_name] = values
    
    return result_df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'