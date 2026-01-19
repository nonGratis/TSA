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
    print("\n[MODE] DEEP LEARNING (Hybrid: Linear Trend + LSTM Residuals)")
    import neural as nn 
    from sklearn.linear_model import LinearRegression
    
    # --- ЕТАП 1: DETRENDING (Виділення тренду) ---
    # Ми вчимо LSTM лише на "хвилях", прибравши глобальний ріст
    print("  [PREP] Виділення глобального тренду...")
    real_data = df_prepared['r_id']
    
    # Підготовка X для регресії (просто індекси часу)
    X_time = np.arange(len(real_data)).reshape(-1, 1)
    y_val = real_data.values.reshape(-1, 1)
    
    # Навчаємо лінійну регресію на всій історії
    trend_model = LinearRegression()
    trend_model.fit(X_time, y_val)
    trend_values = trend_model.predict(X_time).flatten()
    
    # Отримуємо залишки (Residuals) - саме їх буде вчити LSTM!
    # Вони стаціонарні (коливаються навколо нуля), це ідеально для LSTM
    residuals = real_data - trend_values
    residuals.name = 'residuals' # Зберігаємо індекс дат
    
    # --- ЕТАП 2: РОБОТА З LSTM ---
    synth_path = output_dir / 'synthetic_residuals.csv' # Інший файл для синтетики залишків
    model_path = output_dir / 'lstm_hybrid.keras'
    
    # Якщо треба - генеруємо синтетику на основі ЗАЛИШКІВ (а не сирих даних)
    if not synth_path.exists() and not model_path.exists():
        print(f"\n[AUTO-GEN] Генеруємо синтетичні залишки...")
        gen_config = config.copy()
        gen_config['synthetic_length'] = 10000
        gen_config['synthetic_trend'] = 'bootstrap'
        
        # Аналізуємо саме залишки!
        # Створюємо тимчасовий DF
        df_resid = df_prepared.copy()
        df_resid['r_id'] = residuals
        
        analysis_res = mode_analysis(df_resid, config, output_dir)
        # Зберігаємо у спеціальний файл
        orig_synth_func = mode_synthetic # save ref
        
        # Хак: перехоплюємо збереження
        info, _ = mode_synthetic(df_resid, analysis_res, gen_config, output_dir)
        # Перейменовуємо файл, щоб не плутати з основним
        (output_dir / 'synthetic_data.csv').rename(synth_path)

    learner = nn.DeepLearner(window_size=config['dl_window'])
    
    # Завантаження або Навчання
    model_loaded = False
    if not config.get('force_retrain'):
        if learner.load_model(model_path):
            print("  [INFO] Використовуємо попередньо навчену модель.")
            learner.prepare_data(residuals) # Init scaler на залишках
            model_loaded = True
            
    if not model_loaded:
        print("\n[TRAIN] Навчання LSTM на залишках (Residuals)...")
        if synth_path.exists():
            df_synth = pd.read_csv(synth_path)
            # Створюємо дати для синтетики
            start_date = pd.to_datetime('2020-01-01')
            df_synth.index = pd.date_range(start=start_date, periods=len(df_synth), freq='D')
            train_series = df_synth['combined']
        else:
            train_series = residuals # Fallback на реальні, якщо синтетики нема

        X_train, y_train = learner.prepare_data(train_series)
        learner.build_lstm_model()
        learner.train(X_train, y_train, epochs=config['dl_epochs'])
        learner.save_model(model_path)

    # --- ЕТАП 3: ВАЛІДАЦІЯ (Reconstruction) ---
    print("\n[FORECAST] Валідація та Реконструкція...")
    k_steps = config['k_steps']
    window = config['dl_window']
    
    # Тест на останніх k кроках
    split_idx = len(residuals) - k_steps
    
    # 1. Прогноз залишків LSTM
    val_data_slice = residuals.iloc[split_idx-window:] 
    X_test, y_test_scaled = learner.prepare_data(val_data_slice, fit_scaler=False)
    resid_pred_scaled = learner.predict(X_test)
    resid_pred = learner.scaler.inverse_transform(resid_pred_scaled.reshape(-1, 1)).flatten()
    
    # 2. Прогноз тренду (Лінійний)
    # Індекси для тестового періоду
    X_test_time = np.arange(split_idx, len(real_data)).reshape(-1, 1)
    trend_pred = trend_model.predict(X_test_time).flatten()
    
    # 3. Сума (Реконструкція)
    final_pred = trend_pred[:len(resid_pred)] + resid_pred
    actual_y = real_data.iloc[split_idx:].values[:len(final_pred)]
    
    # Метрики
    rmse = mt.calculate_rmse(actual_y, final_pred)
    mae = mt.calculate_mae(actual_y, final_pred)
    mape = mt.calculate_percent_divergence(actual_y, final_pred)
    print(f"  [METRICS] RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    # --- ЕТАП 4: ЕКСТРАПОЛЯЦІЯ (Майбутнє) ---
    # 1. LSTM прогнозує майбутні хвилі
    last_window_series = residuals.iloc[-window:]
    last_window_features = learner.prepare_forecast_input(last_window_series)
    last_date = residuals.index[-1]
    
    future_residuals = learner.extrapolate(last_window_features, start_date=last_date, steps=k_steps)
    
    if np.isnan(future_residuals).any():
        future_residuals = np.nan_to_num(future_residuals)

    # 2. Регресія прогнозує майбутній тренд
    last_time_idx = len(real_data)
    future_time_idx = np.arange(last_time_idx, last_time_idx + k_steps).reshape(-1, 1)
    future_trend = trend_model.predict(future_time_idx).flatten()
    
    # 3. Сума
    future_final = future_trend + future_residuals

    # 4. Повний прогноз для графіку (історія + майбутнє)
    # Проганяємо LSTM по всій історії
    X_full, _ = learner.prepare_data(residuals, fit_scaler=False)
    full_resid_pred = learner.predict(X_full).flatten()
    
    # Відновлюємо повну історію (тренд + прогноз залишків)
    # Увага: full_resid_pred коротший на window_size
    full_trend = trend_values[window:]
    reconstructed_history = full_trend + full_resid_pred

    try:
        dv.plot_lstm_forecast(
            real_series=real_data,
            predictions=reconstructed_history, # Це "навчена" історія
            future_pred=future_final,          # Це прогноз
            window_size=window,
            rmse=rmse,
            save_path=output_dir / 'lstm_hybrid_forecast.svg',
            title="Hybrid Deep Learning (Linear Trend + LSTM Seasonality)"
        )
        print(f"  [PLOT] {output_dir / 'lstm_hybrid_forecast.svg'}")
    except Exception as e:
        print(f"  [ERROR] Візуалізація: {e}")
    import neural as nn 
    
    synth_path = output_dir / 'synthetic_data.csv'
    model_path = output_dir / 'lstm_model_v2.keras'
    
    # 1. Синтетика
    if not synth_path.exists():
        print(f"\n[AUTO-GEN] Генеруємо синтетичні дані...")
        gen_config = config.copy()
        gen_config['synthetic_length'] = 10000
        gen_config['synthetic_trend'] = 'bootstrap'
        analysis_res = mode_analysis(df_prepared, config, output_dir)
        mode_synthetic(df_prepared, analysis_res, gen_config, output_dir)
    
    learner = nn.DeepLearner(window_size=config['dl_window'])
    
    # 2. Навчання / Завантаження
    model_loaded = False
    if not config.get('force_retrain'):
        if learner.load_model(model_path):
            print("  [INFO] Використовуємо попередньо навчену модель.")
            learner.prepare_data(df_prepared['r_id']) # Init scaler
            model_loaded = True
            
    if not model_loaded:
        print("\n[TRAIN] Навчання на синтетиці з календарем...")
        df_synth = pd.read_csv(synth_path)
        start_date = pd.to_datetime('2020-01-01')
        df_synth.index = pd.date_range(start=start_date, periods=len(df_synth), freq='D')
        
        X_train, y_train = learner.prepare_data(df_synth['combined'])
        learner.build_lstm_model()
        learner.train(X_train, y_train, epochs=config['dl_epochs'])
        learner.save_model(model_path)

    # 3. Валідація
    print("\n[FORECAST] Валідація на реальних даних...")
    real_data = df_prepared['r_id']
    k_steps = config['k_steps']
    window = config['dl_window']
    
    # Тест
    split_idx = len(real_data) - k_steps
    val_data_slice = real_data.iloc[split_idx-window:] 
    
    X_test, y_test_scaled = learner.prepare_data(val_data_slice, fit_scaler=False)
    val_pred = learner.predict(X_test)
    
    y_test_real = learner.scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    val_pred_flat = val_pred.flatten()
    
    # Метрики
    min_len = min(len(y_test_real), len(val_pred_flat))
    y_test_real = y_test_real[:min_len]
    val_pred_flat = val_pred_flat[:min_len]
    
    rmse = mt.calculate_rmse(y_test_real, val_pred_flat)
    mae = mt.calculate_mae(y_test_real, val_pred_flat)
    mape = mt.calculate_percent_divergence(y_test_real, val_pred_flat)
    
    print(f"  [METRICS] RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    # 4. ЕКСТРАПОЛЯЦІЯ (ВИПРАВЛЕНО)
    # Беремо останнє вікно
    last_window_series = real_data.iloc[-window:]
    
    # --- FIX: Використовуємо новий метод, який не шукає 'y' ---
    last_window_features = learner.prepare_forecast_input(last_window_series)
    # -----------------------------------------------------------
    
    last_date = real_data.index[-1]
    future_pred = learner.extrapolate(last_window_features, start_date=last_date, steps=k_steps)
    
    if np.isnan(future_pred).any():
        print("  [WARN] NaN у прогнозі! Фікс...")
        future_pred = np.nan_to_num(future_pred, nan=real_data.iloc[-1])

    # 5. Візуалізація
    X_full, _ = learner.prepare_data(real_data, fit_scaler=False)
    full_predictions = learner.predict(X_full)

    try:
        dv.plot_lstm_forecast(
            real_series=real_data,
            predictions=full_predictions.flatten(),
            future_pred=future_pred,
            window_size=window,
            rmse=rmse,
            save_path=output_dir / 'lstm_embeddings_forecast.svg'
        )
        print(f"  [PLOT] {output_dir / 'lstm_embeddings_forecast.svg'}")
    except Exception as e:
        print(f"  [ERROR] Візуалізація: {e}")
        
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