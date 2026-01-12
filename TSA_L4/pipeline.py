import numpy as np
import pandas as pd
from typing import Dict

import data_handler as dh
from kalman import AlphaBetaFilter, estimate_noise_parameters
from adaptive import NISAdapter

def estimate_initial_velocity(y: np.ndarray) -> float:
    """
    Робастна оцінка початкової швидкості.
    Використовує медіану перших різниць для зменшення впливу шуму на старті.
    """
    if len(y) < 3: return 0.0
    # Беремо перші 10 точок
    subset = y[:min(10, len(y))]
    # Видаляємо NaN
    subset = subset[np.isfinite(subset)]
    if len(subset) < 2: return 0.0
    
    diffs = np.diff(subset)
    return float(np.median(diffs))

def select_best_model(y: np.ndarray) -> tuple[int, str]:
    """Вибір моделі: CV (2) або CA (3). Повертає (dim, причина)."""
    y_clean = y[np.isfinite(y)]
    if len(y_clean) < 10: return 2, 'default'
    
    diffs = np.diff(y_clean)
    
    # Якщо процес переважно монотонний (мало падінь), це CV
    negative_diffs = np.sum(diffs < 0)
    if negative_diffs < 0.05 * len(diffs):
        return 2, 'monotonic'
        
    # Якщо динаміка складна
    accel = np.diff(diffs)
    var_v = np.var(diffs)
    var_a = np.var(accel)
    
    if var_a > 10 * var_v:
        return 3, 'high-dynamic'
    else:
        return 2, 'standard'

def run_pipeline(df_prepared: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Виконує фільтрацію на вже підготовлених даних.
    
    Args:
        df_prepared: DataFrame після prepare_timeseries() з колонками 'r_id' та 'imputed'
        config: Dict з параметрами фільтрації
    """
    r_id = df_prepared['r_id'].values.astype(float)
    is_imputed = df_prepared['imputed'].values.astype(bool)
    n = len(r_id)
    
    if config.get('state_dim') is not None:
        state_dim = config['state_dim']
        reason = 'manual'
    else:
        state_dim, reason = select_best_model(r_id)
    
    proc_q, meas_r = estimate_noise_parameters(r_id)
    
    # Краща ініціалізація швидкості
    v_init = estimate_initial_velocity(r_id)
    init_state = np.array([r_id[0], v_init, 0.0])
    
    ab_filter = AlphaBetaFilter(
        dt=1.0, 
        state_dim=state_dim,
        process_noise_q=proc_q,
        measurement_noise_r=meas_r,
        init_state=init_state
    )
    
    # Таблиця налаштувань
    lam_calc = (ab_filter.Q / ab_filter.R) * (ab_filter.dt ** 2)
    model_name = 'CV' if state_dim == 2 else 'CA'
    reason_ua = {
        'standard': 'стандарт', 
        'monotonic': 'монотон', 
        'high-dynamic': 'висока-дин', 
        'default': 'дефолт',
        'manual': 'ручний'
    }[reason]
    print(f"\n[НАЛАШТУВАННЯ ФІЛЬТРА]")
    print(f"  Модель:  {model_name} ({reason_ua})")
    print(f"  Параметри: λ={lam_calc:.4f}  α={ab_filter.alpha:.4f}  β={ab_filter.beta:.4f}")
    print(f"  Шуми:      Q={ab_filter.Q:.1f}  R={ab_filter.R:.1f}")
    print(f"  Початок:   x₀={ab_filter.x:.1f}  v₀={ab_filter.v:.2f}")
    
    adapter = NISAdapter(scale_factor=1.5, decay_factor=0.8) if config.get('adaptive', True) else None
    imputed_mode = config.get('imputed_update_mode', 'skip')
    
    # Loop
    print(f"[ФІЛЬТР] Обробка {n} точок...")
    
    kf_x = np.zeros(n)
    kf_v = np.zeros(n)
    residuals = np.full(n, np.nan)
    alpha_log = np.zeros(n)
    nis_log = np.zeros(n)
    
    kf_x[0] = ab_filter.x
    kf_v[0] = ab_filter.v
    alpha_log[0] = ab_filter.alpha
    
    for i in range(1, n):
        measurement = r_id[i]
        
        # A. PREDICT - ЗАВЖДИ!
        ab_filter.predict()
        
        # B. Check logic
        residual = ab_filter.get_residual(measurement)
        S = ab_filter.get_innovation_variance()
        
        should_update = True
        
        if is_imputed[i]:
            if imputed_mode == 'skip':
                should_update = False
                residual = np.nan 
            elif imputed_mode == 'weighted':
                orig_alpha = ab_filter.alpha
                ab_filter.set_alpha(orig_alpha * 0.05)
                ab_filter.update(measurement)
                ab_filter.set_alpha(orig_alpha)
                should_update = False 
        
        # C. Update & Adapt
        if should_update:
            nis_log[i] = (residual**2) / (S + 1e-9)
            
            if adapter:
                new_q = adapter.update(residual, S, ab_filter.Q, proc_q)
                ab_filter.update_params_from_noise(new_q, meas_r)
                
            ab_filter.update(measurement)
            residuals[i] = residual
            
        kf_x[i] = ab_filter.x
        kf_v[i] = ab_filter.v
        alpha_log[i] = ab_filter.alpha
    
    attrs = {'model_dim': state_dim}
    
    res_df = pd.DataFrame({
        'r_id_raw': r_id,
        'kf_x': kf_x,
        'kf_v': kf_v,
        'residual': residuals,
        'alpha': alpha_log,
        'nis': nis_log,
        'is_imputed': is_imputed,
        'kf_p_var': np.full(n, 0.0),
        'process_q': np.full(n, 0.0),
        'is_anomaly': np.zeros(n, dtype=bool)
    }, index=df_prepared.index)
    
    res_df.attrs = attrs
    return res_df