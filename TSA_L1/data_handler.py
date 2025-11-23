import pandas as pd

def fetch_data(url: str) -> pd.DataFrame:
    
    dfs = pd.read_html(url, header=0, encoding='utf-8')
    if not dfs:
        raise ValueError("No data found.")
    df = dfs[0]
    df.columns = ['timestamp', 'r_id'] + list(df.columns[2:])
    return df[['timestamp', 'r_id']]

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    df_resampled = df.resample('1h').last().ffill().fillna(0)
    return df_resampled