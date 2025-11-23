import pandas as pd

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    df_resampled = df.resample('1h').last().ffill().fillna(0)
    return df_resampled