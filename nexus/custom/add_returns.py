import numpy as np
import pandas as pd

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def add_rets(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    # Assuming your dataframe has columns: ticker, date, adj_close
    df = data.sort_values(['ticker', 'date'])  # Ensure proper sorting

# Calculate log returns within each ticker group
    df['returns'] = df.groupby('ticker')['adj_close'].transform(
        lambda x: np.log(x).diff()
    )

    df['rollmean'] = df.groupby('ticker')['returns'].transform(
        lambda x: x.expanding().mean()
    )      

    # Cumulative sum of squared returns
    df['sumretsq'] = df.groupby('ticker')['returns'].transform(
        lambda x: (x**2).cumsum()
    )
    # Denominator (min of current position or 252)
    df['denom'] = df.groupby('ticker').cumcount() + 1  # +1 because cumcount starts at 0
    df['denom'] = df['denom'].clip(upper=252)

    # Rolling mean over 1 year
    df['rollmean1yr'] = df.groupby('ticker')['returns'].transform(
        lambda x: x.rolling(window=252, min_periods=1).mean()
    )

    # Rolling sum of squared returns over 1 year  
    df['sumretsq1yr'] = df.groupby('ticker')['returns'].transform(
        lambda x: (x**2).rolling(window=252, min_periods=1).sum()
    )

        # Define window sizes (assuming 252 trading days per year)
    windows = {
        '3mo': 63,    # 252/4
        '6mo': 126,   # 252/2
        '9mo': 189,   # 252*3/4
        '1yr': 252,   # 252
        '15mo': 315,  # 252*1.25
        '18mo': 378,  # 252*1.5
        '24mo': 504   # 252*2
        }

    # Calculate rolling sums and denominators for each period
    for period, window_size in windows.items():
        # Rolling sum of returns
        df[f'sumret{period}'] = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).sum()
        )
        
        # Rolling sum of squared returns
        df[f'sumretsq{period}'] = df.groupby('ticker')['returns'].transform(
            lambda x: (x**2).rolling(window=window_size, min_periods=1).sum()
        )
        
        # Denominator (min of current position or window size)
        df[f'denom{period}'] = df.groupby('ticker').cumcount() + 1
        df[f'denom{period}'] = df[f'denom{period}'].clip(upper=window_size)

    # Expanding statistics (already calculated in previous steps)
    df['sumret'] = df.groupby('ticker')['returns'].transform(lambda x: x.cumsum())
    df['sumretsq'] = df.groupby('ticker')['returns'].transform(lambda x: (x**2).cumsum())
    df['_n_'] = df.groupby('ticker').cumcount() + 1

    # Calculate all rolling standard deviations
    def calculate_rolling_std(sumret, sumretsq, denom):
        """Calculate rolling standard deviation using sample formula"""
        mean_ret = sumret / denom
        mean_ret_sq = sumretsq / denom
        variance = mean_ret_sq - mean_ret**2
        # Apply Bessel's correction and take square root
        return np.sqrt((denom / (denom - 1)) * variance)

    # Expanding standard deviation
    df['rollstd'] = calculate_rolling_std(df['sumret'], df['sumretsq'], df['_n_'])

    # Rolling standard deviations for each period
    for period in windows.keys():
        df[f'rollstd{period}'] = calculate_rolling_std(
            df[f'sumret{period}'], 
            df[f'sumretsq{period}'], 
            df[f'denom{period}']
        )
    for col in df.columns:
        print(col)
    return df
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
