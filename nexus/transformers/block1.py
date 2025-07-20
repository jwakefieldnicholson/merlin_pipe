from mage_ai.data_cleaner.transformer_actions.constants import ImputationStrategy
from mage_ai.data_cleaner.transformer_actions.base import BaseAction
from mage_ai.data_cleaner.transformer_actions.constants import ActionType, Axis
from mage_ai.data_cleaner.transformer_actions.utils import build_transformer_action
from pandas import DataFrame
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def add_forward_weights(df, *args, **kwargs):
    """
    Add weight columns for forward-looking analysis periods to handle edge effects.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with daily data and datetime index
    timeframes_months : list, optional
        List of timeframes in months. Default: [3, 6, 9, 12, 15, 18, 24]
    trading_days_per_month : int, optional
        Trading days per month. Default: 21 (252 trading days / 12 months)
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added weight columns (wgt3mos, wgt6mos, etc.)
    """
    trading_days_per_month = kwargs['configuration'].get('trading_days_per_month')
    timeframes_months = kwargs['configuration'].get('timeframes_months')

    if timeframes_months is None:
        timeframes_months = [3, 6, 9, 12, 15, 18, 24]
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Total number of observations
    total_obs = len(df)
    
    # Calculate weights for each timeframe
    for months in timeframes_months:
        # Convert months to trading days
        tau = months * trading_days_per_month
        
        # Create weight column name
        weight_col = f'wgt{months}mos'
        
        # Calculate weights for each observation
        weights = []
        
        for i in range(total_obs):
            # Calculate end point of forward window
            end_point = min(i + tau - 1, total_obs - 1)
            
            # Calculate actual days available vs requested
            days_available = end_point - i + 1
            
            # Weight = actual days / requested days
            weight = days_available / tau
            
            weights.append(weight)
        
        # Add weight column to DataFrame
        result_df[weight_col] = weights
    print(result_df.tail(50))
    return result_df