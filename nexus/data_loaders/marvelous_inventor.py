import io
import pandas as pd
import requests
from pandas import DataFrame
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
yesterday = date.today() - timedelta(days=1)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Intrinio API Key secret
API_KEY = get_secret_value('Intrinio_API')

today = datetime.now()
five_years_ago = (today - relativedelta(years=3)).strftime("%Y-%m-%d")

intrinio.ApiClient().set_api_key(API_KEY) 
intrinio.ApiClient().allow_retries(True)

def get_universe():
# Get today's date minus one day

# Format the date into 'YYYY-MM-DD' string format
    date = yesterday.strftime('%Y-%m-%d')
    #date = '2025-05-14'  # Yesterday's date
    page_size = 100
    start_time = datetime.now()

    # Get Marketcap data with pagination
    marketcap_data = []
    next_page = ''

    while True:
        try:
            response = intrinio.CompanyApi().get_all_companies_daily_metrics(
                on_date=date, 
                page_size=page_size,
                next_page=next_page
            )
            
            print(f'Processing {len(response.daily_metrics)} marketcap entries...')
            
            # Process current page of results
            for daily_metric in response.daily_metrics:
                ticker = daily_metric.company.ticker
                marketcap = daily_metric.market_cap
                
                data = {
                    "ticker" : ticker, 
                    "name": daily_metric.company.name,
                    "date": daily_metric.date.strftime("%Y-%m-%d"),
                    "marketcap": marketcap
                }
                
                marketcap_data.append(data)
            
            # Check if there are more pages
            next_page = response.next_page
            if not next_page:
                break
                
        except ApiException as e:
            print(f"Exception when calling CompanyApi->get_all_companies_daily_metrics: {e}")
            break

    print(f'Found {len(marketcap_data)} marketcap entries for: {date}')
    print(f'Time elapsed: {datetime.now() - start_time}')

    universe = pd.DataFrame(list(marketcap_data))
    selected_universe = universe[((universe.marketcap>2000000000) & (universe.ticker.notna()))]
    
    return selected_universe

def get_eod_data_for_ticker(ticker: str, start_date: str = five_years_ago):
    """
    Get EOD Stock Prices for a single ticker with full pagination
    Returns dictionary of all price data for the ticker
    """
    # Get EOD Stock Prices with pagination
    # https://docs.intrinio.com/documentation/python/get_security_stock_prices_v2
    identifier = ticker
    page_size = 100  # Maximum allowed page size
    total_prices = 0
    ticker_data = {}
    
    try:
        # Initialize pagination
        next_page = ''
        page_count = 0
        max_pages = 1000  # Safety limit
        
        while page_count < max_pages:
            try:
                # Get the current page of results
                response = intrinio.SecurityApi().get_security_stock_prices(
                    identifier, 
                    start_date=start_date, 
                    page_size=page_size, 
                    next_page=next_page
                )
                
                page_count += 1
                security = response.security
                page_prices = len(response.stock_prices)
                total_prices += page_prices
                
                # Process the current page of stock price data
                for stock_price in response.stock_prices:
                    key = f'{ticker}|{stock_price.date}'
                    data = {
                        'security_id': security.id,
                        'company_id': security.company_id,
                        'ticker': security.ticker,
                        'date': stock_price.date,
                        'open': stock_price.open,
                        'high': stock_price.high,
                        'low': stock_price.low,
                        "close": stock_price.close,
                        'adj_open': stock_price.adj_open,
                        'adj_high': stock_price.adj_high,
                        "adj_low": stock_price.adj_low,
                        "adj_close": stock_price.adj_close,
                        'adj_volume': stock_price.adj_volume,
                        'fifty_two_week_high': stock_price.fifty_two_week_high,
                        'fifty_two_week_low': stock_price.fifty_two_week_low,
                        'dividend': stock_price.dividend
                    }
                    ticker_data[key] = data
                
                # Check if there are more pages
                next_page = response.next_page
                if not next_page:
                    break
                    
                # Small delay between pages to be respectful to API
                time.sleep(0.1)
                
            except ApiException as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    logger.warning(f"Rate limit hit for {ticker} on page {page_count}, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    raise e
                    
        if page_count >= max_pages:
            logger.warning(f"Hit maximum page limit for {ticker}, data may be incomplete")
            
        logger.info(f'Found {total_prices} prices for: {ticker} across {page_count} pages')
        return ticker_data
        
    except ApiException as e:
        logger.error(f"Exception when calling SecurityApi->get_security_stock_prices for {ticker}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {e}")
        return {}

def get_eod_data_concurrent(tickers: list, start_date: str = five_years_ago, max_workers: int = 8):
    """
    Get EOD data for multiple tickers concurrently
    """
    logger.info(f"Starting concurrent fetch for {len(tickers)} tickers with {max_workers} workers")
    
    all_results = {}
    failed_tickers = []
    completed_tickers = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(get_eod_data_for_ticker, ticker, start_date): ticker 
            for ticker in tickers
        }
        
        # Use tqdm to show progress
        with tqdm(total=len(tickers), desc="Fetching EOD data") as pbar:
            # Process completed tasks as they finish
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        all_results.update(result)
                        completed_tickers.append(ticker)
                        logger.info(f"Successfully processed {ticker}")
                    else:
                        failed_tickers.append(ticker)
                        logger.warning(f"No data returned for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to process {ticker}: {e}")
                    failed_tickers.append(ticker)
                finally:
                    pbar.update(1)
    
    logger.info(f"Completed: {len(completed_tickers)} tickers")
    logger.info(f"Failed: {len(failed_tickers)} tickers")
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")
    
    return all_results

@data_loader
def load_data_from_api(**kwargs) -> DataFrame:
    start_time = datetime.now()

    selected_universe = get_universe()
    tickers = selected_universe.ticker.tolist()

    print(f'Loading {len(tickers)} tickers') 
    print(tickers)

    # Use concurrent futures to fetch all data
    max_workers = 8  # Adjust based on your API rate limits
    results_dictionary = get_eod_data_concurrent(tickers, five_years_ago, max_workers)
    
    # Convert results to DataFrame
    df = pd.DataFrame(list(results_dictionary.values()))
    
    print(f'Total records fetched: {len(df)}')
    print(f'Unique tickers in final dataset: {len(df.ticker.unique())}')
    print(f'Time elapsed: {datetime.now() - start_time}')
    
    return df

# Optional: Function to retry failed tickers
def retry_failed_tickers(failed_tickers: list, start_date: str = five_years_ago):
    """
    Retry fetching data for failed tickers one by one (no concurrency)
    """
    retry_results = {}
    
    for ticker in failed_tickers:
        print(f"Retrying {ticker}...")
        result = get_eod_data_for_ticker(ticker, start_date)
        if result:
            retry_results.update(result)
            print(f"Successfully retried {ticker}")
        else:
            print(f"Failed again for {ticker}")
    
    return retry_results