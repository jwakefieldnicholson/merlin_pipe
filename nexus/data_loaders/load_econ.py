import io
from fredapi import Fred
import requests
from mage_ai.data_preparation.shared.secrets import get_secret_value
import pandas as pd
from fredapi import Fred
import numpy as np

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# Configuration
FRB_API_KEY = get_secret_value('Fred_API')


# Define what we want to download and their mappings
FRB_RATES = ['DGS10', 'DGS2', 'DGS3', 'DGS1', 'DGS3MO', 'DGS6MO']

FRB_SERIES_MAPPING = {
    'CPIAUCSL': 'CPI',
    'PCUOMFGOMFG': 'ppi', 
    'UNRATE': 'unemprate',
    'CIVPART': 'laborpart',
    'INDPRO': 'indprod',
    'DFF': 'ffr',
    'T10Y3M': 'ycurve10y3m',
    'T10Y2Y': 'ycurve10y2y',
    'DAAA': 'moodycorpbondyield'
}

FRB_INDICES_MAPPING = {
    'SP500': 'snp500'
}


def download_fred_data(series_codes: list, api_key: str = FRB_API_KEY, start_date: str = "2010-01-01") -> pd.DataFrame:
    """Download multiple FRED series and return as DataFrame."""
    fred = Fred(api_key=api_key)
    data_dict = {}
    
    for code in series_codes:
        try:
            print(f"Downloading {code}")
            series = fred.get_series(code)
            filtered_series = series[series.index >= start_date]
            data_dict[code] = filtered_series
        except Exception as e:
            print(f"Error downloading {code}: {e}")
    
    if data_dict:
        df = pd.concat(data_dict, axis=1)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DATE'}, inplace=True)
        return df
    else:
        return pd.DataFrame()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match desired output format."""
    # Rename rate columns (convert to lowercase for some)
    rate_renames = {
        'DGS3MO': 'DGS3mo',
        'DGS6MO': 'DGS6mo'
    }
    
    # Combine all rename mappings
    all_renames = {**rate_renames, **FRB_SERIES_MAPPING, **FRB_INDICES_MAPPING}
    
    return df.rename(columns=all_renames)


def calculate_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived/interpolated variables."""
    df = df.copy()
    
    # Convert DGS values from percentages to decimals
    dgs_cols = ['DGS10', 'DGS2', 'DGS3', 'DGS1', 'DGS3mo', 'DGS6mo']
    for col in dgs_cols:
        if col in df.columns:
            df[col] = df[col] / 100
    
    # Forward fill S&P 500 data
    df['snp500'] = df['snp500'].fillna(method='ffill')
    
    # Interpolated rates (now in decimal form)
    df['dgs1p5'] = (df['DGS1'] + df['DGS2']) / 2  # 1.5-Year Rate
    df['dgs2p5'] = (df['DGS2'] + df['DGS3']) / 2  # 2.5-Year Rate  
    df['dgs9mo'] = (df['DGS6mo'] + df['DGS1']) / 2  # 9-Month Rate
    df['dgs15mo'] = (df['dgs1p5'] + df['DGS1']) / 2  # 15-Month Rate
    
    # S&P 500 log returns (calculated after forward filling)
    df['snp500rets'] = np.log(df['snp500']).diff()
    
    return df

@data_loader
def main() -> pd.DataFrame:
    """Download and process all financial data."""
    print("Downloading FRB rates...")
    rates_df = download_fred_data(FRB_RATES)
    
    print("\nDownloading FRB economic series...")
    series_df = download_fred_data(list(FRB_SERIES_MAPPING.keys()))
    
    print("\nDownloading indices...")
    indices_df = download_fred_data(list(FRB_INDICES_MAPPING.keys()))
    
    # Merge all dataframes
    print("\nMerging data...")
    dataframes = [df for df in [rates_df, series_df, indices_df] if not df.empty]
    
    if not dataframes:
        print("No data downloaded")
        return pd.DataFrame()
    
    # Start with first dataframe and merge others
    merged = dataframes[0].copy()
    for df in dataframes[1:]:
        merged = pd.merge(merged, df, on='DATE', how='outer')
    
    # Rename columns to match desired format
    merged = rename_columns(merged)
    
    # Calculate derived variables
    print("Calculating derived variables...")
    merged = calculate_derived_variables(merged)
    
    # Sort by date
    merged.sort_values('DATE', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    print(f"\nDownload complete! Dataset shape: {merged.shape}")
    print(f"Date range: {merged['DATE'].min()} to {merged['DATE'].max()}")
    
    return merged
