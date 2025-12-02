import pandas as pd

def check_timestamp(df: pd.DataFrame, col: str = 'timestamp', fmt: str = '%d.%m.%Y %H:%M:%S') -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}'")
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')

    bad = df[col].isna().sum()
    if bad:
        print(f"Warning: {bad} rows have invalid '{col}' and will be dropped")
        df = df.dropna(subset=[col])
    return df

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = check_timestamp(df)
    df = df.set_index('timestamp').sort_index()

    col = df.get('r_id')
    if col is None:
        df['r_id'] = pd.Series([], index=df.index, dtype=float)
    else:
        df['r_id'] = pd.to_numeric(col, errors='coerce')

    df = df.dropna(subset=['r_id'])
    df_resampled = df[['r_id']].resample('1H').last().ffill()
    return df_resampled
