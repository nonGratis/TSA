import numpy as np
import pandas as pd

def check_timestamp(df: pd.DataFrame, col: str = 'timestamp') -> pd.DataFrame:
    """Parse timestamps with automatic format detection."""
    if col not in df.columns:
        raise KeyError(f"Відсутня обов'язкова колонка '{col}'")
    
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    bad_count = df[col].isna().sum()
    if bad_count > 0:
        print(f"  Увага: Видалено {bad_count} рядків з некоректним форматом часу.")
        df = df.dropna(subset=[col])
    return df

def _detect_anomalies_cumulative(s: pd.Series) -> pd.Series:
    """Виявляємо падіння лічильника (negative diff)."""
    diffs = s.diff().fillna(0.0)
    mask_drop = diffs < -1e-6
    if mask_drop.any():
        print(f"  Увага: Знайдено {mask_drop.sum()} точок, де лічильник впав.")
    return mask_drop

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports both cumulative and non-cumulative data formats.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Detect data type
    if 'r_id' in df.columns:
        value_col, data_type, freq = 'r_id', 'cumulative', '1h'
    elif 'Subs' in df.columns:
        value_col, data_type, freq = 'Subs', 'non-cumulative', '1D'
    else:
        value_col, data_type, freq = 'r_id', 'cumulative', '1h'
    
    print(f"  Тип даних: {data_type}, частота: {freq}")
    
    # Clean value column - remove all whitespace
    if df[value_col].dtype == 'object':
        df[value_col] = df[value_col].astype(str).str.replace(r'\s+', '', regex=True)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    df = df.dropna(subset=[value_col])
    df = check_timestamp(df)
    
    if df.empty:
        raise ValueError("Дані порожні після очищення!")
    
    df = df.set_index('timestamp').sort_index()
    
    # Statistics
    raw_count = len(df)
    raw_start, raw_end = df.index.min(), df.index.max()

    # Create regular grid
    floor_freq = 'h' if freq == '1h' else 'D'
    start_dt = df.index.min().floor(floor_freq)
    end_dt = df.index.max().ceil(floor_freq)
    regular_grid = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    
    # Interpolate
    combined_index = df.index.union(regular_grid).unique().sort_values()
    s_combined = df[value_col].reindex(combined_index)
    s_interpolated = s_combined.interpolate(method='time', limit_direction='both')
    s_resampled = s_interpolated.reindex(regular_grid)
    
    # Track imputed values
    nearest_idx = df.index.get_indexer(regular_grid, method='nearest')
    nearest_timestamps = df.index[nearest_idx]
    time_diffs = np.abs(regular_grid - nearest_timestamps)
    threshold = pd.Timedelta(days=2) if freq == '1D' else pd.Timedelta(minutes=90)
    imputed_mask = time_diffs > threshold
    
    s_filled = s_resampled.ffill().bfill()
    
    # Check for cumulative anomalies only if cumulative data
    if data_type == 'cumulative':
        anomaly_mask = _detect_anomalies_cumulative(s_filled)
        if anomaly_mask.any():
            s_filled.loc[anomaly_mask] = np.nan
            s_filled = s_filled.ffill()
            imputed_mask = imputed_mask | anomaly_mask

    res_count = len(s_filled)
    res_start, res_end = s_filled.index.min(), s_filled.index.max()

    print(f"  {'Етап':<15} | {'К-сть':<8} | {'Початок':<19} | {'Кінець':<19}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*19}-+-{'-'*19}")
    print(f"  {'Вхідні':<15} | {raw_count:<8} | {str(raw_start):<19} | {str(raw_end):<19}")
    print(f"  {'Оброблені':<15} | {res_count:<8} | {str(res_start):<19} | {str(res_end):<19}")
    print(f"  Імпутовано: {imputed_mask.sum()} ({100.0 * imputed_mask.sum() / res_count:.1f}%)")
    
    return pd.DataFrame({
        'r_id': s_filled.astype(float),
        'imputed': imputed_mask.astype(bool)
    })