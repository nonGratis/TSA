import numpy as np
import pandas as pd
from typing import Optional

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

def _detect_anomalies_series(s: pd.Series, window: int = 24, z: float = 3.0) -> pd.Series:
    """±z*std rolling detector (centered)"""
    mean = s.rolling(window=window, min_periods=1, center=True).mean()
    sd = s.rolling(window=window, min_periods=1, center=True).std(ddof=0).fillna(0.0)
    mask = (s - mean).abs() > (z * sd)
    return mask.fillna(False)

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    - обробка r_id (уникнення None для pd.to_numeric)
    - ресемпл по 1 годині, last() + ffill() (підходить для кумулятивних лічильників)
    - детектор аномалій з заміною аномалії на NaN й ffill
    - повертаємо DataFrame з колонками: 'r_id' та 'imputed' (bool)
    """
    df = check_timestamp(df)
    df = df.set_index('timestamp').sort_index()

    col: Optional[pd.Series] = df.get('r_id')
    if col is None:
        df['r_id'] = pd.Series(index=df.index, dtype=float)
    else:
        df['r_id'] = pd.to_numeric(col, errors='coerce')

    df = df.dropna(subset=['r_id'])
    s = df['r_id'].resample('1h').last()

    imputed = s.isna()

    # r_id істотно кумулятивний лічильник)
    s = s.ffill()

    mask = _detect_anomalies_series(s)
    if mask.any():
        s.loc[mask] = np.nan
        s = s.ffill()
        imputed = imputed | mask.fillna(False)

    df_resampled = pd.DataFrame({'r_id': s, 'imputed': imputed.astype(bool)})
    return df_resampled
