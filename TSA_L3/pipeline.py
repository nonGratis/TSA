import numpy as np
import pandas as pd
from typing import Dict, Optional

import data_handler as dh
from kalman import KalmanFilter, estimate_noise_parameters
from adaptive import AdaptiveQ


def run_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Пайплайн обробки кумулятивних часових рядів з Kalman-фільтром
    
    Етапи:
    1. Очищення/імпутація через prepare_timeseries
    2. Ініціалізація Kalman-фільтра та адаптивного Q
    3. Фільтрація з адаптивною коригуванням Q
    4. Збір результатів
    
    Args:
        df: Сирий DataFrame з колонками 'timestamp' та 'r_id'
        config: Конфігурація з параметрами:
            - state_dim: int (2 або 3) - розмірність стану
            - dt: float - крок дискретизації (год)
            - process_noise: float | None - початковий Q
            - measurement_noise: float | None - R
            - adaptive: bool - використовувати адаптивний Q
            - adaptive_window: int - розмір вікна для AdaptiveQ
            - q_min: float - мінімальний Q
            - q_max: float - максимальний Q
            - adapt_rate: float - швидкість адаптації
            - imputed_update_mode: str - режим роботи з імпутованими ('skip', 'weighted', 'normal')
            - k_steps: int - кількість кроків для прогнозу
    
    Returns:
        DataFrame з колонками:
            - r_id_raw: оригінальні значення
            - r_id_imputed: значення після імпутації
            - is_imputed: bool - чи було імпутовано
            - is_anomaly: bool - чи була аномалія (падіння лічильника)
            - kf_x: оцінка позиції Kalman-фільтром
            - kf_v: оцінка швидкості
            - kf_a: оцінка прискорення (якщо state_dim=3)
            - residual: залишок (measurement - prediction)
            - q_value: поточне значення Q (якщо adaptive=True)
    """
    print("\n=== ПАЙПЛАЙН ОБРОБКИ ЧАСОВИХ РЯДІВ ===\n")
    
    # --- Етап 1: Очищення та імпутація ---
    print("[1/4] Очищення та імпутація даних...")
    df_prepared = dh.prepare_timeseries(df)
    
    n_total = len(df_prepared)
    n_imputed = int(df_prepared['imputed'].sum())
    
    print(f"  Всього точок: {n_total}")
    print(f"  Імпутовано: {n_imputed} ({100.0 * n_imputed / n_total:.1f}%)")
    
    # Зберігаємо оригінальні дані
    r_id_raw = np.asarray(df_prepared['r_id'].values, dtype=float)
    is_imputed = np.asarray(df_prepared['imputed'].values, dtype=bool)
    
    # Визначаємо аномалії (падіння лічильника)
    diffs = np.diff(r_id_raw, prepend=r_id_raw[0])
    is_anomaly = diffs < -1e-6
    n_anomalies = int(is_anomaly.sum())
    print(f"  Аномалії: {n_anomalies}")
    
    # --- Етап 2: Ініціалізація Kalman-фільтра ---
    print("\n[2/4] Ініціалізація Kalman-фільтра...")
    
    state_dim = config.get('state_dim', 2)
    dt = config.get('dt', 1.0)
    
    # Оцінка шумів
    if config.get('process_noise') is None or config.get('measurement_noise') is None:
        print("  Автоматична робастна оцінка параметрів шуму...")
        proc_noise, meas_noise = estimate_noise_parameters(
            r_id_raw, 
            dt=dt, 
            robust=config.get('robust_estimation', True)
        )
        
        if config.get('process_noise') is None:
            process_noise = proc_noise
        else:
            process_noise = config['process_noise']
            
        if config.get('measurement_noise') is None:
            measurement_noise = meas_noise
        else:
            measurement_noise = config['measurement_noise']
    else:
        process_noise = config['process_noise']
        measurement_noise = config['measurement_noise']
    
    print(f"  state_dim: {state_dim}")
    print(f"  dt: {dt}")
    print(f"  Q (початковий): {process_noise:.2e}")
    print(f"  R: {measurement_noise:.2e}")
    
    # Ініціалізація стану з першого виміру
    if state_dim == 2:
        init_state = np.array([r_id_raw[0], 0.0])
    else:
        init_state = np.array([r_id_raw[0], 0.0, 0.0])
    
    kf = KalmanFilter(
        dt=dt,
        state_dim=state_dim,
        process_noise_q=process_noise,
        measurement_noise_r=measurement_noise,
        init_state=init_state
    )
    
    # Адаптивний Q
    use_adaptive = config.get('adaptive', True)
    adaptive_q = None
    
    if use_adaptive:
        print("\n  Адаптивний Q: активовано")
        adaptive_q = AdaptiveQ(
            window=config.get('adaptive_window', 24),
            q_min=config.get('q_min', 1e-6),
            q_max=config.get('q_max', 1e2),
            adapt_rate=config.get('adapt_rate', 1.2),
            init_q=process_noise
        )
    else:
        print("\n  Адаптивний Q: вимкнено")
    
    # --- Етап 3: Фільтрація ---
    print("\n[3/4] Фільтрація часового ряду...")
    
    imputed_mode = config.get('imputed_update_mode', 'skip')
    print(f"  Режим обробки імпутованих даних: {imputed_mode}")
    
    # Масиви для результатів
    kf_x = np.zeros(n_total)
    kf_v = np.zeros(n_total)
    kf_a = np.zeros(n_total) if state_dim >= 3 else None
    residuals = np.zeros(n_total)
    q_values = np.zeros(n_total) if use_adaptive else None
    innovation_covs = np.zeros(n_total)  # Для діагностики
    
    # Перший крок
    kf_x[0] = kf.get_position()
    kf_v[0] = kf.get_velocity()
    if state_dim >= 3 and kf_a is not None:
        kf_a[0] = kf.get_acceleration()
    
    # Основний цикл фільтрації
    for i in range(1, n_total):
        measurement = r_id_raw[i]
        
        # Predict step (завжди)
        kf.predict()
        
        # Residual та інноваційна коваріація до update
        residual = kf.get_residual(measurement)
        innovation_cov = kf.get_innovation_covariance()
        innovation_covs[i] = innovation_cov  # Зберігаємо для діагностики
        
        # Визначаємо, чи робити update
        should_update = True
        update_weight = 1.0
        
        if is_imputed[i]:
            # Обробка імпутованих даних
            if imputed_mode == 'skip':
                # Не робимо update, тільки predict
                should_update = False
            elif imputed_mode == 'weighted':
                # Update з підвищеним R (низька довіра)
                update_weight = 10.0
            # else: 'normal' - звичайний update
        
        # Update step
        if should_update:
            if update_weight != 1.0:
                original_R = kf.R.copy()
                kf.R = kf.R * update_weight
                kf.update(measurement)
                kf.R = original_R
            else:
                kf.update(measurement)
        
        # Зберігаємо результати
        kf_x[i] = kf.get_position()
        kf_v[i] = kf.get_velocity()
        if state_dim >= 3 and kf_a is not None:
            kf_a[i] = kf.get_acceleration()
        
        # Записуємо residual тільки якщо робили update
        if should_update:
            residuals[i] = residual
            
            # Адаптація Q тільки для валідних вимірювань
            if use_adaptive and adaptive_q is not None:
                new_q = adaptive_q.update(residual, innovation_cov)
                kf.update_process_noise(new_q)
                if q_values is not None:
                    q_values[i] = new_q
        else:
            # Для пропущених update залишок не має сенсу
            residuals[i] = 0.0
            if q_values is not None:
                q_values[i] = adaptive_q.q_current if adaptive_q else process_noise
    
    print(f"  Фільтрація завершена: {n_total} точок оброблено")
    
    # --- Діагностика інновацій ---
    print("\n  Діагностика інновацій (контрольний тест):")
    
    # Обчислюємо тільки на валідних вимірах
    valid_residuals = residuals[~is_imputed]
    valid_innov_covs = innovation_covs[~is_imputed]
    
    if len(valid_residuals) > 1:
        empirical_var_innov = float(np.var(valid_residuals, ddof=1))
        mean_S_k = float(np.mean(valid_innov_covs))
        
        print(f"    Емпірична var(інновацій): {empirical_var_innov:.2e}")
        print(f"    Середнє S_k (H P H^T + R): {mean_S_k:.2e}")
        ratio = empirical_var_innov / mean_S_k if mean_S_k > 0 else 0.0
        print(f"    Співвідношення: {ratio:.3f}")
        
        if 0.5 < ratio < 2.0:
            print("    ✓ Фільтр добре налаштовано (співвідношення ≈ 1)")
        elif ratio > 2.0:
            print(f"    ⚠ Емпірична варіація >> S_k: можливо Q або R занадто малі")
            print(f"       Рекомендація: збільшити --q-max або --measurement-noise")
        else:
            print(f"    ⚠ Емпірична варіація << S_k: можливо Q або R занадто великі")
    
    # Статистика адаптації
    if use_adaptive and adaptive_q is not None:
        stats = adaptive_q.get_statistics()
        print("\n  Статистика адаптивного Q:")
        print(f"    Q фінальний: {stats['q_current']:.2e}")
        print(f"    Збільшень Q: {stats['increase_count']}")
        print(f"    Зменшень Q: {stats['decrease_count']}")
        if stats['max_reached_count'] > 0:
            print(f"    Досягнення q_max: {stats['max_reached_count']} разів")
    
    # --- Етап 4: Формування результату ---
    print("\n[4/4] Формування результатів...")
    
    result_data = {
        'r_id_raw': r_id_raw,
        'r_id_imputed': r_id_raw.copy(),  # Після prepare_timeseries вже імпутовано
        'is_imputed': is_imputed,
        'is_anomaly': is_anomaly,
        'kf_x': kf_x,
        'kf_v': kf_v,
        'residual': residuals,
        'valid_measurement': ~is_imputed  # Додаємо маску валідних вимірів
    }
    
    if state_dim >= 3:
        result_data['kf_a'] = kf_a
    
    if use_adaptive:
        result_data['q_value'] = q_values
    
    result_df = pd.DataFrame(result_data, index=df_prepared.index)
    
    print(f"  DataFrame створено: {len(result_df)} рядків, {len(result_df.columns)} колонок")
    print("\n=== ПАЙПЛАЙН ЗАВЕРШЕНО ===\n")
    
    return result_df


def predict_k_steps_ahead(kf: KalmanFilter, k: int) -> np.ndarray:
    """
    Прогноз k кроків вперед для поточного стану фільтра
    
    Args:
        kf: Налаштований Kalman-фільтр
        k: Кількість кроків
        
    Returns:
        Масив прогнозів розміром k
    """
    return kf.predict_k_steps(k)
