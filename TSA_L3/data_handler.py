import numpy as np
import pandas as pd

def check_timestamp(df: pd.DataFrame, col: str = 'timestamp', fmt: str = '%d.%m.%Y %H:%M:%S') -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Відсутня обов'язкова колонка '{col}'")
    
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
        
    bad_mask = df[col].isna()
    bad_count = bad_mask.sum()
    if bad_count > 0:
        print(f"  Увага: Видалено {bad_count} рядків з некоректним форматом часу.")
        df = df.dropna(subset=[col])
    return df

def _detect_anomalies_cumulative(s: pd.Series) -> pd.Series:
    """Виявляємо падіння лічильника (negative diff)."""
    diffs = s.diff().fillna(0.0).astype(float)
    mask_drop: pd.Series = diffs < -1e-6
    if mask_drop.any():
        print(f"  Увага: Знайдено {mask_drop.sum()} точок, де лічильник впав (скидання?).")
    return mask_drop

def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Дедуплікація (keep='last') для секунд з кількома подіями.
    2. Розширення сітки до floor('h') для початку від 0
    3. Time-weighted інтерполяція.
    """
    n_total = len(df)
    df = check_timestamp(df)
    
    df = df.set_index('timestamp').sort_index()
    duplicates = df.index.duplicated(keep='last')
    n_dupes = duplicates.sum()
    if n_dupes > 0:
        print(f"  Увага: Високочастотні події: {n_dupes} секунд мали кілька оновлень. Зберігання останнього стану для кожного.")
        df = df[~duplicates]

    # Очистка r_id
    col = df.get('r_id')
    if col is None:
        df['r_id'] = np.nan
    else:
        df['r_id'] = pd.to_numeric(col, errors='coerce')
    
    df = df.dropna(subset=['r_id'])
    
    if df.empty:
        raise ValueError("Дані порожні! Перевірте файл.")

    print(f"  Часовий проміжок: {df.index.min()} -> {df.index.max()}")

    start_dt = df.index.min().floor('h')
    end_dt = df.index.max().ceil('h')
    
    regular_grid = pd.date_range(start=start_dt, end=end_dt, freq='1h')
    
    # Інтерполяція
    combined_index = df.index.union(regular_grid).unique().sort_values()
    s_combined = df['r_id'].reindex(combined_index)
    
    # limit_direction='both' дозволить заповнити 19:00 значенням з 19:32 (backward fill для старту)
    s_interpolated = s_combined.interpolate(method='time', limit_direction='both')
    
    s_resampled = s_interpolated.reindex(regular_grid)
    

    # Імпутація (Gap Detection), якщо реальних даних не було +/- 90 хв, вважаємо це імпутованим
    nearest_idx = df.index.get_indexer(regular_grid, method='nearest')
    nearest_timestamps = df.index[nearest_idx]
    time_diffs = np.abs(regular_grid - nearest_timestamps)
    imputed_mask = time_diffs > pd.Timedelta(minutes=90)
    s_filled = s_resampled.ffill().bfill() # Гарантія відсутності NaN
    
    # Аномалії
    anomaly_mask = _detect_anomalies_cumulative(s_filled)
    if anomaly_mask.any():
        s_filled.loc[anomaly_mask] = np.nan
        s_filled = s_filled.ffill()
        imputed_mask = imputed_mask | anomaly_mask

    print(f"  Створено {len(s_filled)} годинних точок (Початок: {start_dt})")
    
    return pd.DataFrame({
        'r_id': s_filled.astype(float),
        'imputed': imputed_mask.astype(bool)
    })