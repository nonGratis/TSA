import numpy as np
import pandas as pd
from typing import Dict, Optional

import data_handler as dh
from kalman import AlphaBetaFilter, estimate_noise_parameters
# FIX: Імпортуємо новий адаптер NIS замість старого AdaptiveAlpha
from adaptive import NISAdapter

def run_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    print("\n=== ПАЙПЛАЙН: ALPHA-BETA ===\n")

    # --- 1. Data Prep ---
    print("[1/4] Очищення...")
    df_prepared = dh.prepare_timeseries(df)
    n_total = len(df_prepared)
    
    r_id_input = np.asarray(df_prepared['r_id'].values, dtype=float)
    is_imputed = np.asarray(df_prepared['imputed'].values, dtype=bool)
    
    print("[2/4] Ініціалізація...")
    state_dim = int(config.get('state_dim', 2))
    dt = float(config.get('dt', 1.0))
    
    proc_noise, meas_noise = estimate_noise_parameters(r_id_input)
    
    cfg_q = config.get('process_noise')
    cfg_r = config.get('measurement_noise')
    
    Q_base = float(cfg_q) if cfg_q is not None else float(proc_noise)
    R_fixed = float(cfg_r) if cfg_r is not None else float(meas_noise)
    
    print(f"  Base Q: {Q_base:.2e}, Fixed R: {R_fixed:.2e}")
    
    init_state = np.zeros(state_dim)
    init_state[0] = r_id_input[0]
    
    ab_filter = AlphaBetaFilter(
        dt=dt, 
        state_dim=state_dim, 
        process_noise_q=Q_base, 
        measurement_noise_r=R_fixed,
        init_state=init_state
    )
    
    use_adaptive = config.get('adaptive', True)
    adapter = NISAdapter(dof=1) if use_adaptive else None
    if use_adaptive: print("  NIS Adaptation: ON")
    
    print("\n[3/4] Фільтрація...")
    
    kf_x = np.zeros(n_total)
    kf_v = np.zeros(n_total)
    kf_a = np.zeros(n_total) if state_dim == 3 else None
    kf_p_var = np.zeros(n_total)
    
    residuals = np.full(n_total, np.nan)
    alpha_vals = np.zeros(n_total)
    nis_vals = np.zeros(n_total)
    
    kf_x[0] = ab_filter.get_position()
    kf_p_var[0] = ab_filter.get_position_variance()
    alpha_vals[0] = ab_filter.alpha
    residuals[0] = 0.0
    
    imputed_mode = config.get('imputed_update_mode', 'skip')
    
    for i in range(1, n_total):
        measurement = r_id_input[i]
        
        # A. Prediction
        ab_filter.predict()
        
        # B. Innovation & Theoretical Variance
        residual = ab_filter.get_residual(measurement)
        S = ab_filter.get_innovation_variance()
        
        should_update = True
        
        # C. Handling Imputed
        if is_imputed[i]:
            if imputed_mode == 'skip':
                should_update = False
                residual = np.nan 
            elif imputed_mode == 'weighted':
                # Manual intervention, adaptation might be unstable here
                old_alpha = ab_filter.alpha
                ab_filter.set_alpha(old_alpha * 0.1) 
                ab_filter.update(measurement)
                ab_filter.set_alpha(old_alpha)
                should_update = False 
        
        # D. Adaptation & Update
        current_nis = 0.0
        if should_update:
            current_nis = (residual**2) / (S + 1e-9)
            
            if adapter:
                # 1. Adapt Q based on NIS
                # FIX: Використовуємо метод update нового адаптера
                new_Q = adapter.update(residual, S, ab_filter.Q, Q_base)
                
                # 2. Recalculate Filter Gains (Alpha/Beta)
                ab_filter.update_params_from_noise(new_Q, R_fixed)
            
            ab_filter.update(measurement)
        
        # E. Store
        residuals[i] = residual
        kf_x[i] = ab_filter.get_position()
        kf_v[i] = ab_filter.get_velocity()
        kf_p_var[i] = ab_filter.get_position_variance()
        alpha_vals[i] = ab_filter.alpha
        nis_vals[i] = current_nis
        
        if kf_a is not None:
            kf_a[i] = ab_filter.get_acceleration()
            
    print(f"  Оброблено {n_total} точок.")

    result_data = {
        'r_id_raw': r_id_input,
        'is_imputed': is_imputed,
        'is_anomaly': np.zeros_like(is_imputed),
        'kf_x': kf_x,
        'kf_v': kf_v,
        'residual': residuals,
        'kf_p_var': kf_p_var,
        'q_value': alpha_vals,
        'nis': nis_vals,
        'valid_measurement': ~is_imputed
    }
    if kf_a is not None:
        result_data['kf_a'] = kf_a
        
    return pd.DataFrame(result_data, index=df_prepared.index)