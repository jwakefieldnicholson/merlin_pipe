import io
import pandas as pd
import requests

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from mage_ai.data_preparation.shared.secrets import get_secret_value

#from __future__ import print_function
import time
import pandas as pd
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import numpy as np

class MarketCapFetcher:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        Initialize the MarketCapFetcher
        
        Args:
            api_key: Your Intrinio API key
            max_workers: Maximum number of concurrent threads (be mindful of API rate limits)
        """
        self.api_key = api_key
        self.max_workers = max_workers
        
        # Configure Intrinio SDK
        intrinio.ApiClient().set_api_key(api_key)
        intrinio.ApiClient().allow_retries(True)
        self.company_api = intrinio.CompanyApi()
    
    def get_marketcap_for_ticker(self, ticker: str, start_date: str = '2018-01-01', 
                                end_date: str = '', frequency: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Fetch market cap data for a single ticker, handling pagination to get all results
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD), empty for latest
            frequency: Data frequency ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
            
        Returns:
            DataFrame with market cap data or None if failed
        """
        try:            
            all_data = []
            next_page = ''
            page_count = 0
            max_pages = 1000  # Safety limit to prevent infinite loops
            
            while page_count < max_pages:
                try:
                    response = self.company_api.get_company_historical_data(
                        identifier=ticker,
                        tag='marketcap',
                        frequency=frequency,
                        type='',
                        start_date=start_date,
                        end_date=end_date,
                        sort_order='desc',
                        page_size=100,
                        next_page=next_page
                    )
                    
                    page_count += 1
                    
                    # Process the current page data
                    if hasattr(response, 'historical_data') and response.historical_data:
                        for item in response.historical_data:
                            all_data.append({
                                'ticker': ticker,
                                'date': item.date,
                                'marketcap': item.value
                            })
                                            
                    # Check if there are more pages
                    if hasattr(response, 'next_page') and response.next_page:
                        next_page = response.next_page
                        # Add a small delay between pages to be respectful to the API
                        time.sleep(0.1)
                    else:
                        # No more pages, break the loop
                        break
                        
                except ApiException as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        print(f"Rate limit hit for {ticker} on page {page_count}, waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    else:
                        raise e
            
            if page_count >= max_pages:
                print(f"Hit maximum page limit for {ticker}, data may be incomplete")
            
            # Convert all data to DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                print(f"Successfully fetched {len(df)} total records for {ticker} across {page_count} pages")
                return df
            else:
                print(f"No market cap data found for {ticker}")
                return None
                
        except ApiException as e:
            print(f"API error for {ticker}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error for {ticker}: {e}")
            return None
    
    def get_marketcap_concurrent(self, tickers: List[str], start_date: str = '2018-01-01',
                                end_date: str = '', frequency: str = 'daily') -> pd.DataFrame:
        """
        Fetch market cap data for multiple tickers concurrently
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            frequency: Data frequency
            
        Returns:
            Combined DataFrame with market cap data for all tickers
        """
        print(f"Starting concurrent fetch for {len(tickers)} tickers with {self.max_workers} workers")
        
        all_data = []
        failed_tickers = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.get_marketcap_for_ticker, ticker, start_date, end_date, frequency): ticker 
                for ticker in tickers
            }
            
            # Process completed tasks
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_data.append(result)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    print(f"Failed to process {ticker}: {e}")
                    failed_tickers.append(ticker)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Successfully fetched data for {len(combined_df['ticker'].unique())} tickers")
        else:
            print("No data was successfully fetched")
            combined_df = pd.DataFrame(columns=['ticker', 'date', 'marketcap'])
        
        if failed_tickers:
            print(f"Failed to fetch data for tickers: {failed_tickers}")
        
        return combined_df
    
    def merge_with_panel_data(self, panel_df: pd.DataFrame, marketcap_df: pd.DataFrame,
                             ticker_col: str = 'ticker', date_col: str = 'date') -> pd.DataFrame:
        """
        Merge market cap data with existing panel data
        
        Args:
            panel_df: Your existing panel dataframe
            marketcap_df: Market cap dataframe from API
            ticker_col: Name of ticker column in panel_df
            date_col: Name of date column in panel_df
            
        Returns:
            Merged dataframe
        """
        print("Merging market cap data with panel data")
        
        # Ensure date columns are datetime
        if date_col in panel_df.columns:
            panel_df[date_col] = pd.to_datetime(panel_df[date_col])
        marketcap_df['date'] = pd.to_datetime(marketcap_df['date'])
        
        # Merge on ticker and date
        merged_df = panel_df.merge(
            marketcap_df[['ticker', 'date', 'marketcap']], 
            left_on=[ticker_col, date_col],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Drop duplicate ticker column if it was created
        if 'ticker' in merged_df.columns and ticker_col != 'ticker':
            merged_df = merged_df.drop('ticker', axis=1)
        
        print(f"Merged dataframe shape: {merged_df.shape}")
        return merged_df

@data_loader
def load_data_from_api(data, *args, **kwargs):
    """
    Template for loading data from API
    """
    API_KEY = get_secret_value('Intrinio_API')
    fetcher = MarketCapFetcher(API_KEY, max_workers=5)  # Adjust max_workers based on your API rate limits
    
    # Get unique tickers from your panel data
    unique_tickers = data['ticker'].unique().tolist()
    
    # Fetch market cap data concurrently
    marketcap_data = fetcher.get_marketcap_concurrent(
        tickers=unique_tickers,
        start_date='2022-01-01',
        end_date='2025-7-31',
        frequency='daily'
    )
    
    # Merge with your panel data
    final_df = fetcher.merge_with_panel_data(data, marketcap_data)
    
    print("Final dataframe with market cap:")
    print(final_df.head())

    return final_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'