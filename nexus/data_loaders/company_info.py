import io
import pandas as pd
from datetime import datetime
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from mage_ai.data_preparation.shared.secrets import get_secret_value
API_KEY = get_secret_value('Intrinio_API')
intrinio.ApiClient().set_api_key(API_KEY)
intrinio.ApiClient().allow_retries(True)

@data_loader
def get_minimal_sector_info(data):
    """
    Get just the essential sector/industry information
    Returns a more focused dataset
    """
    unique_tickers = data['ticker'].unique()
    
    print(f'Processing minimal sector information for {len(unique_tickers)} unique tickers')
    
    def get_minimal_info(ticker):
        try:
            response = intrinio.CompanyApi().get_company(ticker)
            return {
                'ticker': ticker,
                'company_name': response.name,
                'sector': getattr(response, 'sector', None),
                'industry_category': getattr(response, 'industry_category', None),
                'industry_group': getattr(response, 'industry_group', None)
            }
        except ApiException as e:
            print(f"Error fetching minimal info for {ticker}: {e}")
            return {
                'ticker': ticker,
                'company_name': None,
                'sector': None,
                'industry_category': None,
                'industry_group': None
            }
    
    results_list = []
    start_time = datetime.now()
    
    # Process tickers sequentially
    for i, ticker in enumerate(unique_tickers):
        print(f'Processing {i+1}/{len(unique_tickers)}: {ticker}')
        result = get_minimal_info(ticker)
        results_list.append(result)
        
        # Optional: Add a small delay to be respectful to the API
        # time.sleep(0.1)
    
    sector_df = pd.DataFrame(results_list)
    merged_df = data.merge(sector_df, on='ticker', how='left')
    
    current_time = datetime.now()
    print(f'Minimal sector processing completed in: {current_time - start_time}')
    return merged_df
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'