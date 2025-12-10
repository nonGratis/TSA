import numpy as np
import pandas as pd

def check_timestamp(df: pd.DataFrame, col: str = 'timestamp', fmt: str = '%d.%m.%Y %H:%M:%S') -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}'")
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
    bad = df[col].isna().sum()
    if bad:
        print(f"Попередження: {bad} рядків мають невірний '{col}' і будуть видалені")
        df = df.dropna(subset=[col])
    return df

def _detect_anomalies_cumulative(s: pd.Series) -> pd.Series:
    """Виявляємож тільки падіння лічильника (аналогічно до diff() < 0)"""
    diffs = s.diff().fillna(0.0).astype(float)
    mask_drop: pd.Series = diffs < -1e-6
    
    return mask_drop

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    - обробка r_id (уникнення None для pd.to_numeric)
    - ресемпл по 1 годині, last() + ffill() (підходить для кумулятивних лічильників)
    - детектор аномалій з заміною аномалії на NaN й ffill
    - повертаємо DataFrame з колонками: 'r_id' та 'imputed' (bool)
    """
    df = check_timestamp(df)
    df = df.set_index('timestamp').sort_index()

    col = df.get('r_id')
    if col is None:
        df['r_id'] = np.nan
    else:
        df['r_id'] = pd.to_numeric(col, errors='coerce')

    df = df.dropna(subset=['r_id'])
    s_resampled = df['r_id'].resample('1h').last()
    imputed_mask = s_resampled.isna()

    # ffill значення не змінювалося
    s_filled = s_resampled.ffill()
    
    anomaly_mask = _detect_anomalies_cumulative(s_filled)

    if anomaly_mask.any():
        s_filled.loc[anomaly_mask] = np.nan
        s_filled = s_filled.ffill()
        imputed_mask = imputed_mask | anomaly_mask

    return pd.DataFrame({
        'r_id': s_filled,
        'imputed': imputed_mask.astype(bool)
    })
