import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import data_loader as dl
import data_handler as dh
import data_vizer as dv
import pipeline as pl
import metrics as mt
from kalman import AlphaBetaFilter
from regression import RegressionForecaster 
from forecasting import ClassicalForecaster
import forecasting as fc

import decomposition as dec
import properties as prop
import clustering as clust
import synthetic as synth
import regression as reg
import selector as sel
import neural as nn

def parse_arguments():
    """Парсинг аргументів командного рядка."""
    parser = argparse.ArgumentParser(
        description='Time Series Analysis: Kalman фільтрація + Поглиблений аналіз',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:

  # Базова фільтрація
  python main.py --file data.csv --mode filtering
  
  # Повний аналіз (декомпозиція + властивості + кластеризація)
  python main.py --file data.csv --mode analysis
  
  # Генерація синтетичних даних
  python main.py --file data.csv --mode synthetic
  
  # Все разом
  python main.py --file data.csv --mode full
        """
    )
    
    # Основні параметри
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Шлях до CSV файлу')
    group.add_argument('--url', type=str, help='URL до Google Sheet')
    
    # Режим роботи
    parser.add_argument('--mode', type=str, default='full',
                        choices=['filtering', 'analysis', 'synthetic', 'forecasting', 'arima-grid', 'ma-grid', 'full', 'auto-select', 'regression', 'deep-learning'],
                        help='Режим роботи: filtering, analysis, synthetic, forecasting, arima-grid, ma-grid, full, auto-select, regression, deep-learning')
    
    # Параметри фільтрації
    parser.add_argument('--state-dim', type=int, default=None, choices=[2, 3],
                       help='Розмірність фільтра (2=CV, 3=CA). Авто якщо не вказано')
    parser.add_argument('--adaptive', action='store_true', default=True,
                       help='Увімкнути NIS адаптацію Q')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
    parser.add_argument('--imputed-mode', dest='imputed_update_mode', 
                       type=str, default='skip', choices=['skip', 'weighted'])
    parser.add_argument('--k-steps', type=int, default=12,
                       help='Кількість кроків прогнозування')
    
    # Параметри декомпозиції
    parser.add_argument('--decomp-period', type=int, default=None,
                       help='Період сезонності для декомпозиції (авто якщо не вказано)')
    parser.add_argument('--decomp-seasonal', type=int, default=7,
                       help='Розмір вікна сезонності (має бути непарним)')
    parser.add_argument('--diff-data', action='store_true',
                       help='Диференціювати дані перед аналізом (data.diff())')
    
    # Параметри кластеризації
    parser.add_argument('--cluster-method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical'],
                       help='Метод кластеризації')
    parser.add_argument('--n-clusters', type=int, default=3,
                       help='Кількість кластерів (для kmeans, hierarchical)')
    parser.add_argument('--cluster-window', type=int, default=24,
                       help='Розмір вікна для кластеризації')
    parser.add_argument('--cluster-features', type=str, default='statistical',
                       choices=['raw', 'statistical'],
                       help='Тип ознак для кластеризації')
    
    # Параметри синтетичних даних
    parser.add_argument('--synthetic-length', type=int, default=1000,
                       help='Довжина синтетичного ряду')
    parser.add_argument('--synthetic-seed', type=int, default=42,
                       help='Random seed для синтетичних даних')
    parser.add_argument('--synthetic-trend', type=str, default='polynomial',
                       choices=['linear', 'polynomial', 'exponential', 'logarithmic', 'bootstrap'],
                       help='Тип тренду для синтетичних даних')
    parser.add_argument('--synthetic-poly-degree', type=int, default=2,
                       help='Степінь полінома для polynomial тренду (1-5)')
    
    # Параметри для прогнозування
    parser.add_argument('--ma-window', type=int, default=24, help='Вікно для Moving Average')
    parser.add_argument('--ma-windows', type=str, default='7,14,24,48,96', help='Вікна для MA grid search')
    parser.add_argument('--arima-order', type=str, default='1,1,1', help='ARIMA order p,d,q')
    parser.add_argument('--arima-p-max', type=int, default=2, help='Max p для ARIMA grid search')
    parser.add_argument('--arima-d-max', type=int, default=1, help='Max d для ARIMA grid search')
    parser.add_argument('--arima-q-max', type=int, default=2, help='Max q для ARIMA grid search')
    
    parser.add_argument('--poly-degree', type=int, default=2, help='Ступінь полінома для регресії')
    
    # Параметри нейромережі
    parser.add_argument('--dl-epochs', type=int, default=50, 
                       help='Кількість епох навчання (default: 50)')
    parser.add_argument('--dl-window', type=int, default=60, 
                       help='Розмір вікна (Lookback) для LSTM (default: 60)')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Примусове перенавчання моделі (ігнорувати збережений файл)')   
    
    # Виведення
    parser.add_argument('--output-dir', type=str, default='images',
                       help='Директорія для збереження графіків')
    
    return parser.parse_args()

def mode_deep_learning(df_prepared, config, output_dir):
    """Deep learning mode with clean separation of concerns."""
    print("\n[MODE] DEEP LEARNING (LSTM для Time Series)")
    
    # Завантаження даних для навчання
    try:
        synth_path = output_dir / 'synthetic_data.csv'
        df_synth = pd.read_csv(synth_path)
        data_train = df_synth['combined']
        print(f"  [INFO] Завантажено {len(data_train)} синтетичних записів для навчання")
    except:
        print("  [WARN] Синтетика не знайдена. Використовуємо реальні дані...")
        data_train = df_prepared['r_id']

    # Ініціалізація та підготовка
    learner = nn.DeepLearner(window_size=config['dl_window'])
    X_train, y_train = learner.prepare_data(data_train)
    print(f"  Тензори сформовано: X={X_train.shape}, y={y_train.shape}")
    
    # Навчання моделі
    learner.build_lstm_model()
    history = learner.train(X_train, y_train, epochs=config['dl_epochs'])
    
    # Валідація на реальних даних
    real_series = df_prepared['r_id']
    X_real, y_real = learner.prepare_data(real_series)
    predictions = learner.predict(X_real)
    
    # Метрики
    real_trimmed = real_series.values[config['dl_window']:]
    rmse = mt.calculate_rmse(real_trimmed, predictions.flatten())
    print(f"\n[RESULT] LSTM RMSE на реальних даних: {rmse:.2f}")
    
    # Екстраполяція
    last_window_scaled = learner.scaler.transform(
        real_series.values[-config['dl_window']:].reshape(-1, 1)
    )
    future_pred = learner.extrapolate(last_window_scaled, steps=config['k_steps'])
    
    # Візуалізація (делегована в data_vizer)
    save_path = str(output_dir / 'lstm_forecast.svg')
    dv.plot_lstm_forecast(
        real_series=real_series,
        predictions=predictions,
        future_pred=future_pred,
        window_size=config['dl_window'],
        rmse=rmse,
        title='Deep Learning (LSTM) Forecast',
        save_path=save_path
    )
    
    print(f"  Графік збережено: {save_path}")

def mode_filtering(df_raw, df_prepared, config, output_dir):   
    df_res = pl.run_pipeline(df_prepared, config)
    
    selected_dim = df_res.attrs.get('model_dim', 2)
    
    mask_valid = ~np.isnan(df_res['residual'])
    if mask_valid.sum() > 2:
        metrics_result = mt.evaluate_filter_performance(
            np.asarray(df_res['r_id_raw'][mask_valid].values, dtype=float),
            np.asarray(df_res['kf_x'][mask_valid].values, dtype=float),
            np.asarray(df_res['residual'][mask_valid].values, dtype=float)
        )
    else:
        print("  [WARN] Замало даних для метрик")
        metrics_result = {}
    
    print(f"\n[ПРОГНОЗ НА {config['k_steps']} КРОКІВ]")
    
    last_alpha = float(df_res['alpha'].iloc[-1]) if 'alpha' in df_res else 0.1
    last_x = float(df_res['kf_x'].iloc[-1])
    last_v = float(df_res['kf_v'].iloc[-1])
    
    init_state = [last_x, last_v]
    if selected_dim == 3 and 'kf_a' in df_res.columns:
        init_state.append(float(df_res['kf_a'].iloc[-1]))
    
    est_std = metrics_result.get('residual_std', 1.0)
    pred_r = float(est_std**2)
    if pred_r <= 0:
        pred_r = 1.0
    
    pred_filter = AlphaBetaFilter(
        dt=1.0,
        state_dim=selected_dim,
        init_state=np.array(init_state),
        measurement_noise_r=pred_r
    )
    
    preds, vars_pred = pred_filter.predict_k_steps(config['k_steps'])
    stds = np.sqrt(vars_pred)
    
    print(f"  +1 крок:        {preds[0]:.2f}")
    print(f"  +{config['k_steps']} кроків:     {preds[-1]:.2f} ± {1.96*stds[-1]:.2f} (95% ДІ)")
    
    dv.plot_data_preprocessing(
        df_raw, df_res,
        title="Етап 0: Попередня обробка даних",
        save_path=str(output_dir / '00_data_prep.svg')
    )
    
    dv.plot_kalman_results(
        df_res, (preds, vars_pred),
        title=f"Етап 1: Kalman Filter (Dim={selected_dim})",
        save_path=str(output_dir / '01_kf_results.svg')
    )
    
    dv.plot_residuals_analysis(
        df_res,
        title="Етап 2: Діагностика залишків",
        save_path=str(output_dir / '02_kf_diagnostics.svg')
    )
        
    return df_res, metrics_result


def mode_analysis(df_prepared, config, output_dir):
    data_series = df_prepared['r_id']
    
    # Диференціювання якщо вказано
    if config.get('diff_data', False):
        data_series = data_series.diff().dropna()
        print(f"\n[ДИФЕРЕНЦІЮВАННЯ]")
        print(f"  Застосовано data.diff()")
        print(f"  Нова довжина: {len(data_series)}")
    
    decomposer = dec.TimeSeriesDecomposer(
        period=config.get('decomp_period'),
        seasonal=config.get('decomp_seasonal', 7),
        robust=True
    )
    
    decomp_result = decomposer.decompose(data_series)
    decomp_stats = decomposer.get_statistics()
    
    print(f"\n[ДЕКОМПОЗИЦІЯ РЯДУ]")
    print(f"  Період:           {decomp_stats['period']} (авто)" if config.get('decomp_period') is None 
          else f"  Період:           {decomp_stats['period']} (задано)")
    print(f"  Сезонне вікно:    {decomposer.seasonal}")
    print(f"  Сила тренду:      {decomp_stats['trend_strength']:.3f}")
    print(f"  Сила сезонності:  {decomp_stats['seasonal_strength']:.3f}")
    
    dv.plot_decomposition(
        decomp_result,
        title="STL Декомпозиція часового ряду",
        save_path=str(output_dir / '03_decomposition.svg')
    )
    
    analyzer = prop.TimeSeriesProperties()
    props_result = analyzer.analyze_all(data_series, nlags=40)
    
    dv.plot_stationarity_tests(
        data_series,
        props_result['stationarity'],
        save_path=str(output_dir / '04_stationarity.svg'),
        window_size=config.get('cluster_window')
    )
    
    dv.plot_hurst_and_acf(
        data_series,
        props_result,
        save_path=str(output_dir / '05_hurst_acf.svg')
    )
    
    clusterer = clust.TimeSeriesClusterer(
        method=config.get('cluster_method', 'kmeans'),
        n_clusters=config.get('n_clusters', 3)
    )
    
    cluster_result = clusterer.cluster(
        data_series,
        window_size=config.get('cluster_window'),
        feature_type=config.get('cluster_features', 'statistical')
    )
    
    silhouette = clusterer.calculate_silhouette_score(
        cluster_result['features'],
        cluster_result['labels']
    )
    print(f"  Якість (silhouette): {silhouette:.3f}")
    
    dv.plot_clustering_results(
        data_series,
        cluster_result,
        save_path=str(output_dir / '06_clustering.svg')
    )
    
    
    components_df = pd.DataFrame({
        'original': decomp_result['observed'].values,
        'trend': decomp_result['trend'].values,
        'seasonal': decomp_result['seasonal'].values,
        'resid': decomp_result['resid'].values
    })
    
    corr_matrix = components_df.corr()
    print(f"\n[КОРЕЛЯЦІЇ КОМПОНЕНТІВ]")
    print(f"  Тренд ↔ Оригінал:     {corr_matrix.loc['trend', 'original']:.3f}")
    print(f"  Сезон ↔ Оригінал:     {corr_matrix.loc['seasonal', 'original']:.3f}")
    print(f"  Залишки ↔ Сезон:      {corr_matrix.loc['resid', 'seasonal']:.3f}")
    
    return {
        'decomposition': decomp_result,
        'decomposition_stats': decomp_stats,
        'properties': props_result,
        'clustering': cluster_result,
        'correlation': corr_matrix
    }


def mode_synthetic(df_prepared, analysis_result, config, output_dir):
    data_series = df_prepared['r_id']
    
    if analysis_result is None:
        print("\n  [INFO] Спочатку виконується аналіз для отримання властивостей...")
        analyzer = prop.TimeSeriesProperties()
        props_result = analyzer.analyze_all(data_series, nlags=40)
        
        decomposer = dec.TimeSeriesDecomposer(period=config.get('decomp_period'))
        decomp_result = decomposer.decompose(data_series)
    else:
        props_result = analysis_result['properties']
        decomp_result = analysis_result['decomposition']
    
    # Генеруємо синтетичні дані з довжиною як у реальних даних (якщо не вказано інше)
    synth_length = config.get('synthetic_length')
    if synth_length is None:
        synth_length = len(data_series)
        print(f"\n  [INFO] Використовується довжина реальних даних: {synth_length}")
    
    generator = synth.SyntheticTimeSeriesGenerator(
        length=synth_length,
        random_state=config.get('synthetic_seed', 42)
    )
    
    synthetic_combined, synthetic_info = generator.generate_from_real_properties(
        data_series,
        decomp_result,
        props_result,
        trend_type=config.get('synthetic_trend', 'polynomial'),
        poly_degree=config.get('synthetic_poly_degree', 2)
    )
    
    synthetic_series = pd.Series(synthetic_combined)
    synth_analyzer = prop.TimeSeriesProperties()
    synth_props = synth_analyzer.analyze_all(synthetic_series, nlags=40)
    
    real_h = props_result['hurst'].get('hurst', np.nan)
    synth_h = synth_props['hurst'].get('hurst', np.nan)
    
    print(f"\n[ПОРІВНЯННЯ: РЕАЛЬНІ vs СИНТЕТИЧНІ]")
    print(f"  Параметр      Реальні      Синтетичні")
    print(f"  {'─'*42}")
    print(f"  Hurst (H)     {real_h:<12.3f} {synth_h:<12.3f}")
    print(f"  Середнє (μ)   {data_series.mean():<12.1f} {synthetic_series.mean():<12.1f}")
    print(f"  Std (σ)       {data_series.std():<12.1f} {synthetic_series.std():<12.1f}")
    
    dv.plot_synthetic_vs_real(
        data_series,
        synthetic_combined,
        save_path=str(output_dir / '07_synthetic_comparison.svg')
    )
    
    synth_df = pd.DataFrame({
        'combined': synthetic_info['combined'],
        'trend': synthetic_info['trend'],
        'seasonal': synthetic_info['seasonal'],
        'noise': synthetic_info['noise']
    })
    
    synth_path = output_dir / 'synthetic_data.csv'
    synth_df.to_csv(synth_path, index=False)
    
    return synthetic_info, synth_props

def mode_forecasting(df_prepared, config, output_dir):
    print("РЕЖИМ: FORECASTING & COMPARISON (Група вимог 1 & 2)")

    data_series = df_prepared['r_id'].astype(float)
    k_steps = config['k_steps']
    
    # 1. Розділення на train/test для валідації (останні k_steps)
    train = data_series.iloc[:-k_steps]
    test = data_series.iloc[-k_steps:]
    
    print(f"\n[ВАЛІДАЦІЯ МОДЕЛЕЙ]")
    print(f"  Train size: {len(train)}")
    print(f"  Test size:  {len(test)}")
    
    forecaster = fc.ClassicalForecaster()
    results = {}
    
    # --- A. Moving Average ---
    ma_pred, _ = forecaster.moving_average_forecast(train, window=config['ma_window'], steps=k_steps)
    results['MA'] = ma_pred
    
    # --- B. Holt-Winters (Exponential Smoothing) ---
    # Визначаємо сезонність
    seasonal_period = config.get('decomp_period') if config.get('decomp_period') else 24
    hw_pred, _ = forecaster.holt_winters_forecast(train, steps=k_steps, seasonal_periods=seasonal_period)
    results['Holt-Winters'] = hw_pred
    
    # --- C. ARIMA ---
    # Парсинг ордера
    try:
        p, d, q = map(int, config['arima_order'].split(','))
        order = (p, d, q)
    except:
        order = (1, 1, 1)
        
    arima_pred, _ = forecaster.arima_forecast(train, order=order, steps=k_steps)
    results['ARIMA'] = arima_pred
    
    # --- D. Kalman Filter (Alpha-Beta) ---
    # Швидке налаштування фільтра на train
    ab_filter = AlphaBetaFilter(dt=1.0, state_dim=2) # CV модель як базова
    # "Прогрів" фільтра
    for val in train.values:
        ab_filter.predict()
        ab_filter.update(val)
    
    kf_pred, _ = ab_filter.predict_k_steps(k_steps)
    results['Kalman (AB)'] = kf_pred
    
    # --- Порівняння ---
    print(f"\n{'Model':<15} | {'RMSE':<10} | {'MAE':<10} | {'MAPE (%)':<10}")
    print("-" * 55)
    
    best_model = None
    best_rmse = float('inf')
    
    for name, pred in results.items():
        rmse = mt.calculate_rmse(test.values, pred)
        mae = mt.calculate_mae(test.values, pred)
        mape = mt.calculate_percent_divergence(test.values, pred) # Це по суті MAPE
        
        print(f"{name:<15} | {rmse:<10.2f} | {mae:<10.2f} | {mape:<10.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name
            
    print(f"\n  Переможець за RMSE: {best_model}")
    
    # --- Екстраполяція (на майбутнє) ---
    print(f"\n[ЕКСТРАПОЛЯЦІЯ НА МАЙБУТНЄ]")
    # Тренуємо на ВСІХ даних
    print(f"  Прогноз на {k_steps} (1.0 інтервал), {int(k_steps*1.5)} (1.5), {k_steps*2} (2.0) кроків...")
    
    final_steps = k_steps * 2
    
    # Holt-Winters як найбільш ймовірний переможець для сезонних даних
    hw_future, hw_conf = forecaster.holt_winters_forecast(data_series, steps=final_steps, seasonal_periods=seasonal_period)
    
    # Kalman
    ab_filter_full = AlphaBetaFilter(dt=1.0, state_dim=2)
    for val in data_series.values:
        ab_filter_full.predict()
        ab_filter_full.update(val)
    kf_future, kf_vars = ab_filter_full.predict_k_steps(final_steps)
    kf_conf = np.column_stack([kf_future - 1.96*np.sqrt(kf_vars), kf_future + 1.96*np.sqrt(kf_vars)])

    if not config.get('no_plots'):
        dv.plot_forecast_validation(
            train, test, results, k_steps,
            title="Порівняльна валідація моделей прогнозування часових рядів",
            save_path=str(output_dir / '08_forecast_validation.svg')
        )
        
        dv.plot_forecast_extrapolation(
            data_series, hw_future, hw_conf, kf_future, kf_conf, k_steps,
            title=f"Екстраполяція часового ряду: порівняння методів прогнозування (k={final_steps})",
            save_path=str(output_dir / '09_forecast_extrapolation.svg')
        )
        
        print(f"  Графіки порівняння збережено у {output_dir}")

    return results


def mode_arima_grid(df_prepared, config, output_dir):
    """ARIMA Grid Search: перебір всіх комбінацій (p,d,q)."""
    print("РЕЖИМ: ARIMA GRID SEARCH")
    
    data_series = df_prepared['r_id'].astype(float)
    k_steps = config['k_steps']
    
    train = data_series.iloc[:-k_steps]
    test = data_series.iloc[-k_steps:]
    
    p_max = config.get('arima_p_max', 2)
    d_max = config.get('arima_d_max', 1)
    q_max = config.get('arima_q_max', 2)
    
    print(f"\n[GRID SEARCH]")
    print(f"  p: 0..{p_max}, d: 0..{d_max}, q: 0..{q_max}")
    print(f"  Train: {len(train)}, Test: {len(test)}")
    
    forecaster = fc.ClassicalForecaster()
    results = []
    
    print(f"\n{'Order':<12} | {'RMSE':<10} | {'MAE':<10} | {'AIC':<12}")
    print("-" * 50)
    
    for p in range(p_max + 1):
        for d in range(d_max + 1):
            for q in range(q_max + 1):
                order = (p, d, q)
                try:
                    pred, conf = forecaster.arima_forecast(train, order=order, steps=k_steps)
                    
                    # Перевіряємо чи прогноз валідний (не всі нулі)
                    if np.allclose(pred, 0):
                        raise ValueError("Empty forecast")
                    
                    rmse = mt.calculate_rmse(test.values, pred)
                    mae = mt.calculate_mae(test.values, pred)
                    
                    # AIC з fitted моделі
                    aic = forecaster.models.get('arima').aic if forecaster.models.get('arima') else float('inf')
                    
                    results.append({
                        'order': order,
                        'p': p, 'd': d, 'q': q,
                        'rmse': rmse,
                        'mae': mae,
                        'aic': aic,
                        'pred': pred
                    })
                    
                    print(f"({p},{d},{q}){'':<6} | {rmse:<10.2f} | {mae:<10.2f} | {aic:<12.2f}")
                    
                except Exception as e:
                    print(f"({p},{d},{q}){'':<6} | {'FAIL':<10} | {'':<10} | {str(e)[:20]}")
    
    if not results:
        print("\n  [ERROR] Жодна модель не підійшла!")
        return None
    
    # Найкращі моделі
    best_rmse = min(results, key=lambda x: x['rmse'])
    best_aic = min(results, key=lambda x: x['aic'])
    
    print(f"\n{'='*50}")
    print(f"  Найкраща за RMSE: ARIMA{best_rmse['order']} (RMSE={best_rmse['rmse']:.2f})")
    print(f"  Найкраща за AIC:  ARIMA{best_aic['order']} (AIC={best_aic['aic']:.2f})")
    
    # Зберігаємо результати
    results_df = pd.DataFrame([{
        'p': r['p'], 'd': r['d'], 'q': r['q'],
        'rmse': r['rmse'], 'mae': r['mae'], 'aic': r['aic']
    } for r in results])
    results_df.to_csv(output_dir / 'arima_grid_results.csv', index=False)
    print(f"\n  Результати збережено: {output_dir / 'arima_grid_results.csv'}")
    
    return {
        'results': results,
        'best_rmse': best_rmse,
        'best_aic': best_aic
    }


def mode_ma_grid(df_prepared, config, output_dir):
    """MA Grid Search: перебір різних розмірів вікна."""
    print("РЕЖИМ: MA WINDOW GRID SEARCH")
    
    data_series = df_prepared['r_id'].astype(float)
    k_steps = config['k_steps']
    
    train = data_series.iloc[:-k_steps]
    test = data_series.iloc[-k_steps:]
    
    # Парсинг вікон
    windows = [int(w) for w in config.get('ma_windows', '7,14,24,48,96').split(',')]
    
    print(f"\n[GRID SEARCH]")
    print(f"  Вікна: {windows}")
    print(f"  Train: {len(train)}, Test: {len(test)}")
    
    forecaster = fc.ClassicalForecaster()
    results = []
    
    print(f"\n{'Window':<10} | {'RMSE':<10} | {'MAE':<10} | {'MAPE (%)':<10}")
    print("-" * 50)
    
    for w in windows:
        try:
            pred, _ = forecaster.moving_average_forecast(train, window=w, steps=k_steps)
            
            if np.allclose(pred, 0):
                raise ValueError("Empty forecast")
            
            rmse = mt.calculate_rmse(test.values, pred)
            mae = mt.calculate_mae(test.values, pred)
            mape = mt.calculate_percent_divergence(test.values, pred)
            
            results.append({
                'window': w,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'pred': pred
            })
            
            print(f"{w:<10} | {rmse:<10.2f} | {mae:<10.2f} | {mape:<10.2f}")
            
        except Exception as e:
            print(f"{w:<10} | {'FAIL':<10} | {'':<10} | {str(e)[:20]}")
    
    if not results:
        print("\n  [ERROR] Жодне вікно не підійшло!")
        return None
    
    # Найкращі
    best_rmse = min(results, key=lambda x: x['rmse'])
    best_mape = min(results, key=lambda x: x['mape'])
    
    print(f"\n{'='*50}")
    print(f"  Найкраще за RMSE: window={best_rmse['window']} (RMSE={best_rmse['rmse']:.2f})")
    print(f"  Найкраще за MAPE: window={best_mape['window']} (MAPE={best_mape['mape']:.2f}%)")
    
    # Зберігаємо результати
    results_df = pd.DataFrame([{
        'window': r['window'], 'rmse': r['rmse'], 'mae': r['mae'], 'mape': r['mape']
    } for r in results])
    results_df.to_csv(output_dir / 'ma_grid_results.csv', index=False)
    print(f"\n  Результати збережено: {output_dir / 'ma_grid_results.csv'}")
    
    return {
        'results': results,
        'best_rmse': best_rmse,
        'best_mape': best_mape
    }

def mode_regression(df, config, output_dir):
    print("\n[MODE] Регресійний аналіз: Порівняння моделей регресії з експоненційним згладжуванням")
    
    series = df['r_id']
    k_steps = config['k_steps']
    train = series.iloc[:-k_steps]
    test = series.iloc[-k_steps:]
    print(f"  Train size: {len(train)}, Test size: {k_steps}")
    rc = RegressionForecaster()
    fc = ClassicalForecaster()
    
    results = {}
    
    results['Linear'] = rc.linear_regression(train, k_steps)
    results['Poly(d=2)'] = rc.polynomial_regression(train, k_steps, degree=2)
    results['Poly(d=3)'] = rc.polynomial_regression(train, k_steps, degree=3)
    hw_pred, _ = fc.holt_winters_forecast(train, steps=k_steps, 
                                          seasonal_periods=config.get('decomp_period', 24))
    results['Holt-Winters'] = hw_pred

    # Обчислення метрик
    metrics = {}
    metrics_str = f"{'Model':<15} | {'RMSE':<10}\n" + "-"*30 + "\n"
    
    for name, pred in results.items():
        rmse = mt.calculate_rmse(test.values, pred)
        metrics[name] = rmse
        metrics_str += f"{name:<15} | {rmse:<10.2f}\n"
    
    print(metrics_str)

    # Візуалізація (делегована в data_vizer)
    save_path = str(output_dir / 'regression_comparison.svg')
    dv.plot_regression_comparison(
        train=train,
        test=test,
        predictions=results,
        metrics=metrics,
        k_steps=k_steps,
        title='Регресія проти експоненціального згладжування',
        save_path=save_path
    )
    
    print(f"[PLOT] Графік збережено: {save_path}")


def main():
    args = parse_arguments()
    config = vars(args)

    try:
        if args.file:
            df_raw = pd.read_csv(args.file)
            source = args.file
        else:
            df_raw = dl.fetch_data(args.url)
            source = args.url
        
        if df_raw is None:
            raise ValueError("Не вдалося завантажити дані")
        
        print(f"[ЗАВАНТ.] {source} → {len(df_raw)} рядків")
        
        print("\n[PREP] Обробка даних...")
        df_prepared = dh.prepare_timeseries(df_raw)
        
        output_dir = Path(__file__).parent / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.mode == 'filtering':
            mode_filtering(df_raw, df_prepared, config, output_dir)
            
        elif args.mode == 'analysis':
            mode_analysis(df_prepared, config, output_dir)
            
        elif args.mode == 'synthetic':
            analysis_result = mode_analysis(df_prepared, config, output_dir)
            mode_synthetic(df_prepared, analysis_result, config, output_dir)
            
        elif args.mode == 'forecasting':
            mode_forecasting(df_prepared, config, output_dir)
        
        elif args.mode == 'arima-grid':
            mode_arima_grid(df_prepared, config, output_dir)
        
        elif args.mode == 'ma-grid':
            mode_ma_grid(df_prepared, config, output_dir)
            
        elif args.mode == 'full':            
            df_filtered, metrics_result = mode_filtering(df_raw, df_prepared, config, output_dir)
            analysis_result = mode_analysis(df_prepared, config, output_dir)
            synthetic_info, synth_props = mode_synthetic(
                df_prepared, analysis_result, config, output_dir
            )
            mode_forecasting(df_prepared, config, output_dir)
        elif args.mode == 'regression':
            mode_regression(df_prepared, config, output_dir)
        
        elif args.mode == 'auto-select':
            # Запуск нашого Selector
            print("РЕЖИМ: АВТОМАТИЧНИЙ ВИБІР МЕТОДУ (Група вимог 3)")
            selector = sel.ModelSelector(df_prepared['r_id'], freq_period=config['decomp_period'] or 24)
            best_model_name, best_forecast = selector.select_best_model(k_steps=config['k_steps'])
            print(f"\n[RESULT] Переможець алгоритму: {best_model_name}")
        elif args.mode == 'deep-learning':
            mode_deep_learning(df_prepared, config, output_dir)
                
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("ПОМИЛКА")
        print("="*60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()