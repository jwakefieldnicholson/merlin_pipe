# merge_deduplicate_flexible_block.py
import pandas as pd

@transformer
def merge_and_deduplicate_flexible(*dataframes_to_merge, **kwargs):
    """
    Merges data from an arbitrary number of upstream transformer blocks
    and drops duplicates.

    Args:
        *dataframes_to_merge: A variable number of pd.DataFrame objects
                              passed as positional arguments from upstream blocks.
        **kwargs: Standard keyword arguments provided by Mage (e.g., execution_date).

    Returns:
        pd.DataFrame: Merged and deduplicated DataFrame.
    """
    if not dataframes_to_merge:
        print("No DataFrames received for merging. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"Received {len(dataframes_to_merge)} DataFrames for merging.")

    # Log shapes of incoming DataFrames for debugging
    for i, df in enumerate(dataframes_to_merge):
        df['date'] = pd.to_datetime(df['date']).dt.date
        if isinstance(df, pd.DataFrame):
            print(f"  DataFrame {i+1} shape: {df.shape}")
        else:
            print(f"  Warning: Argument {i+1} is not a DataFrame: {type(df)}")

    # 1. Merge the data
    # Use pd.concat to stack all DataFrames
    # It's robust to different column sets, filling missing with NaN
    try:
        merged_df = dataframes_to_merge[0]
        for df in dataframes_to_merge[1:]:
            merged_df = pd.merge(merged_df, df, on=["date", "ticker"], suffixes=('', '_drop'))
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_drop')])
    except Exception as e:
        print(f"Error during concatenation: {e}")
        # Depending on your error strategy, you might want to re-raise or return empty
        return pd.DataFrame()

    # 2. Drop Duplicates
    # You still need to decide what constitutes a "duplicate".
    # Assuming dropping based on all columns for now.
    # Adjust `subset=['col1', 'col2']` if you need specific column-based deduplication.
    initial_rows = merged_df.shape[0]
    deduplicated_df = merged_df.drop_duplicates()
    final_rows = deduplicated_df.shape[0]

    print(f"Shape after deduplication: {deduplicated_df.shape}")
    print(f"Dropped {initial_rows - final_rows} duplicate rows.")
    
    return deduplicated_df