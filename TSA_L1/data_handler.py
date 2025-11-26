import pandas as pd

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')
    df = df.set_index('timestamp')
    
    df = df.sort_index()

    df_resampled = df.resample('1h').last().interpolate(method='linear').fillna(0)
    return df_resampled