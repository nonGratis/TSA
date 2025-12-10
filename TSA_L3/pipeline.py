import numpy as np
import pandas as pd
from typing import Dict, Optional

import data_handler as dh
from kalman import AlphaBetaFilter, estimate_noise_parameters
from adaptive import AdaptiveAlpha

def run_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Пайплайн обробки часових рядів за допомогою Alpha-Beta фільтра.
    """
    print("\n=== ПАЙПЛАЙН: ALPHA-BETA ФІЛЬТР ===\n")

    # --- Етап 1: Очищення ---
    print("[1/4] Очищення та імпутація...")
    df_prepared = dh.prepare_timeseries(df)
    n_total = len(df_prepared)
    
    r_id_raw = np.asarray(df_prepared['r_id'].values, dtype=float)
    is_imputed = np.asarray(df_prepared['imputed'].values, dtype=bool)
    
    # --- Етап 2: Ініціалізація ---
    print("[2/4] Ініціалізація фільтра...")
    state_dim = config.get('state_dim', 2)
    dt = config.get('dt', 1.0)
    
    # Оцінка шумів для розрахунку базових alpha/beta
    proc_noise, meas_noise = estimate_noise_parameters(r_id_raw)
    
    # Оверрайд шумів з конфігу (якщо задано)
    Q = config.get('process_noise') if config.get('process_noise') else proc_noise
    R = config.get('measurement_noise') if config.get('measurement_noise') else meas_noise
    
    print(f"  state_dim: {state_dim}, dt: {dt}")
    print(f"  Ref Q: {Q:.2e}, Ref R: {R:.2e}")
    
    # Створення фільтра
    # Ініціалізація стану: перша точка - x, v=0, a=0
    init_state = np.zeros(state_dim)
    init_state[0] = r_id_raw[0]
    
    ab_filter = AlphaBetaFilter(
        dt=dt, 
        state_dim=state_dim, 
        process_noise_q=Q, 
        measurement_noise_r=R,
        init_state=init_state
    )
    
    print(f"  Initial Params: α={ab_filter.alpha:.4f}, β={ab_filter.beta:.4f}")
    
    # Адаптер
    use_adaptive = config.get('adaptive', True)
    adapter = None
    if use_adaptive:
        print("  Адаптивний режим: ON")
        adapter = AdaptiveAlpha(base_alpha=ab_filter.alpha)
    
    # --- Етап 3: Фільтрація ---
    print("\n[3/4] Фільтрація...")
    
    # Результати
    kf_x = np.zeros(n_total)
    kf_v = np.zeros(n_total)
    kf_a = np.zeros(n_total) if state_dim == 3 else None
    kf_p_var = np.zeros(n_total) # "Variance" proxy
    residuals = np.zeros(n_total)
    alpha_vals = np.zeros(n_total)
    
    # Перша точка
    kf_x[0] = ab_filter.get_position()
    kf_p_var[0] = ab_filter.get_position_variance()
    alpha_vals[0] = ab_filter.alpha
    
    imputed_mode = config.get('imputed_update_mode', 'skip')
    
    for i in range(1, n_total):
        measurement = r_id_raw[i]
        
        # 1. Predict
        ab_filter.predict()
        
        # 2. Update logic
        residual = ab_filter.get_residual(measurement)
        
        should_update = True
        
        if is_imputed[i]:
            if imputed_mode == 'skip':
                should_update = False
            # weighted не має прямого аналогу в simple alpha-beta, 
            # але ми можемо просто зменшити alpha тимчасово
            elif imputed_mode == 'weighted':
                old_alpha = ab_filter.alpha
                ab_filter.set_alpha(old_alpha * 0.1) # low trust
                ab_filter.update(measurement)
                ab_filter.set_alpha(old_alpha) # restore
                should_update = False # вже оновили вручну
        
        if should_update:
            # Адаптація перед оновленням
            if use_adaptive and adapter:
                new_alpha = adapter.update(residual, R)
                ab_filter.set_alpha(new_alpha)
            
            ab_filter.update(measurement)
            residuals[i] = residual
        else:
            residuals[i] = 0.0 # або np.nan
            
        # Збереження
        kf_x[i] = ab_filter.get_position()
        kf_v[i] = ab_filter.get_velocity()
        kf_p_var[i] = ab_filter.get_position_variance()
        alpha_vals[i] = ab_filter.alpha
        
        if state_dim == 3:
            kf_a[i] = ab_filter.get_acceleration()
            
    print(f"  Оброблено {n_total} точок.")
    if use_adaptive and adapter:
        stats = adapter.get_statistics()
        print(f"  Alpha stats: Current={stats['alpha_current']:.4f}, +{stats['increase_count']}, -{stats['decrease_count']}")

    # --- Етап 4: Формування результату ---
    print("\n[4/4] Збір даних...")
    
    result_data = {
        'r_id_raw': r_id_raw,
        'r_id_imputed': r_id_raw, 
        'is_imputed': is_imputed,
        'is_anomaly': np.zeros_like(is_imputed), # placeholder
        'kf_x': kf_x,
        'kf_v': kf_v,
        'residual': residuals,
        'kf_p_var': kf_p_var,
        'q_value': alpha_vals, # Reuse column name for visualizer compatibility (it expects q_value for plot)
        'valid_measurement': ~is_imputed
    }
    if state_dim == 3:
        result_data['kf_a'] = kf_a
        
    return pd.DataFrame(result_data, index=df_prepared.index)