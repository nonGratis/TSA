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

# Нові модулі
import decomposition as dec
import properties as prop
import clustering as clust
import synthetic as synth
import advanced_vizer as adv


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
                       choices=['filtering', 'analysis', 'synthetic', 'full'],
                       help='Режим роботи: filtering, analysis, synthetic, full')
    
    # Параметри фільтрації
    parser.add_argument('--state-dim', type=int, default=None, choices=[2, 3],
                       help='Розмірність фільтра (2=CV, 3=CA). Авто якщо не вказано')
    parser.add_argument('--process-noise', type=float, default=None,
                       help='Process noise Q')
    parser.add_argument('--measurement-noise', type=float, default=None,
                       help='Measurement noise R')
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
    
    # Виведення
    parser.add_argument('--output-dir', type=str, default='images',
                       help='Директорія для збереження графіків')
    parser.add_argument('--no-plots', action='store_true',
                       help='Не зберігати графіки')
    
    return parser.parse_args()


def mode_filtering(df_raw, config, output_dir):
    """Режим: Тільки фільтрація (ЛР1-3)."""
    print("\n" + "="*60)
    print("РЕЖИМ: KALMAN ФІЛЬТРАЦІЯ")
    print("="*60)
    
    # Pipeline фільтрації
    df_res = pl.run_pipeline(df_raw, config)
    
    selected_dim = df_res.attrs.get('model_dim', 2)
    
    # Metrics
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
    
    # Prediction
    print(f"\nПрогноз {config['k_steps']} кроків...")
    
    last_alpha = float(df_res['alpha'].iloc[-1]) if 'alpha' in df_res else 0.1
    last_x = float(df_res['kf_x'].iloc[-1])
    last_v = float(df_res['kf_v'].iloc[-1])
    
    init_state = [last_x, last_v]
    if selected_dim == 3 and 'kf_a' in df_res.columns:
        init_state.append(float(df_res['kf_a'].iloc[-1]))
    
    est_std = metrics_result.get('residual_std', 1.0)
    pred_r = float(config['measurement_noise']) if config['measurement_noise'] else float(est_std**2)
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
    
    print(f"  Next (+1h): {preds[0]:.2f}")
    print(f"  End (+{config['k_steps']}h): {preds[-1]:.2f} ± {1.96*stds[-1]:.2f} (95% CI)")
    
    # Visualize
    if not config.get('no_plots'):
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
        
        print(f"\n  Графіки збережено у {output_dir}")
    
    return df_res, metrics_result


def mode_analysis(df_prepared, config, output_dir):
    """Режим: Поглиблений аналіз (ЛР4)."""
    print("\n" + "="*60)
    print("РЕЖИМ: ПОГЛИБЛЕНИЙ АНАЛІЗ")
    print("="*60)
    
    data_series = df_prepared['r_id']
    
    # 1. Декомпозиція
    print("\n[1/4] ДЕКОМПОЗИЦІЯ...")
    decomposer = dec.TimeSeriesDecomposer(
        period=config.get('decomp_period'),
        seasonal=config.get('decomp_seasonal', 7),
        robust=True
    )
    
    decomp_result = decomposer.decompose(data_series)
    decomp_stats = decomposer.get_statistics()
    
    print(f"\n  Статистика декомпозиції:")
    print(f"    Період: {decomp_stats['period']}")
    print(f"    Сила тренду: {decomp_stats['trend_strength']:.3f}")
    print(f"    Сила сезонності: {decomp_stats['seasonal_strength']:.3f}")
    
    if not config.get('no_plots'):
        adv.plot_decomposition(
            decomp_result,
            title="STL Декомпозиція часового ряду",
            save_path=str(output_dir / '03_decomposition.svg')
        )
    
    # 2. Властивості
    print("\n[2/4] АНАЛІЗ ВЛАСТИВОСТЕЙ...")
    analyzer = prop.TimeSeriesProperties()
    props_result = analyzer.analyze_all(data_series, nlags=40)
    
    if not config.get('no_plots'):
        adv.plot_stationarity_tests(
            data_series,
            props_result['stationarity'],
            save_path=str(output_dir / '04_stationarity.svg'),
            window_size=config.get('cluster_window')
        )
        
        adv.plot_hurst_and_acf(
            data_series,
            props_result,
            save_path=str(output_dir / '05_hurst_acf.svg')
        )
    
    # 3. Кластеризація
    print("\n[3/4] КЛАСТЕРИЗАЦІЯ...")
    clusterer = clust.TimeSeriesClusterer(
        method=config.get('cluster_method', 'kmeans'),
        n_clusters=config.get('n_clusters', 3)
    )
    
    cluster_result = clusterer.cluster(
        data_series,
        window_size=config.get('cluster_window'),
        feature_type=config.get('cluster_features', 'statistical')
    )
    
    # Silhouette score
    silhouette = clusterer.calculate_silhouette_score(
        cluster_result['features'],
        cluster_result['labels']
    )
    print(f"\n  Silhouette Score: {silhouette:.3f}")
    
    if not config.get('no_plots'):
        adv.plot_clustering_results(
            data_series,
            cluster_result,
            save_path=str(output_dir / '06_clustering.svg')
        )
    
    # 4. Кореляційний аналіз
    print("\n[4/4] КОРЕЛЯЦІЙНИЙ АНАЛІЗ...")
    
    # Матриця кореляцій між компонентами
    components_df = pd.DataFrame({
        'original': decomp_result['observed'].values,
        'trend': decomp_result['trend'].values,
        'seasonal': decomp_result['seasonal'].values,
        'resid': decomp_result['resid'].values
    })
    
    corr_matrix = components_df.corr()
    print("\n  Кореляційна матриця компонентів:")
    print(corr_matrix.round(3))
    
    return {
        'decomposition': decomp_result,
        'decomposition_stats': decomp_stats,
        'properties': props_result,
        'clustering': cluster_result,
        'correlation': corr_matrix
    }


def mode_synthetic(df_prepared, analysis_result, config, output_dir):
    """Режим: Генерація синтетичних даних."""
    print("\n" + "="*60)
    print("РЕЖИМ: ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ")
    print("="*60)
    
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
    
    # Генеруємо синтетичні дані
    generator = synth.SyntheticTimeSeriesGenerator(
        length=config.get('synthetic_length', 1000),
        random_state=config.get('synthetic_seed', 42)
    )
    
    synthetic_combined, synthetic_info = generator.generate_from_real_properties(
        data_series,
        decomp_result,
        props_result
    )
    
    # Порівняльний аналіз
    print("\n=== ПОРІВНЯННЯ ВЛАСТИВОСТЕЙ ===\n")
    
    # Створюємо Series для синтетичних даних
    synthetic_series = pd.Series(synthetic_combined)
    
    # Аналіз синтетичних
    print("Аналіз синтетичних даних:")
    synth_analyzer = prop.TimeSeriesProperties()
    synth_props = synth_analyzer.analyze_all(synthetic_series, nlags=40)
    
    # Порівняння
    print("\n" + "="*50)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ:")
    print("="*50)
    
    real_h = props_result['hurst'].get('hurst', np.nan)
    synth_h = synth_props['hurst'].get('hurst', np.nan)
    
    print(f"{'Метрика':<25} | {'Реальні':<15} | {'Синтетичні':<15}")
    print("-"*60)
    print(f"{'Hurst exponent':<25} | {real_h:<15.4f} | {synth_h:<15.4f}")
    print(f"{'Стаціонарність (ADF)':<25} | {props_result['stationarity']['conclusion']:<15} | {synth_props['stationarity']['conclusion']:<15}")
    print(f"{'Середнє':<25} | {data_series.mean():<15.2f} | {synthetic_series.mean():<15.2f}")
    print(f"{'Std':<25} | {data_series.std():<15.2f} | {synthetic_series.std():<15.2f}")
    print(f"{'Значущих ACF лагів':<25} | {props_result['autocorrelation']['n_significant_acf']:<15} | {synth_props['autocorrelation']['n_significant_acf']:<15}")
    print("="*60 + "\n")
    
    # Візуалізація
    if not config.get('no_plots'):
        adv.plot_synthetic_vs_real(
            data_series,
            synthetic_combined,
            save_path=str(output_dir / '07_synthetic_comparison.svg')
        )
    
    # Зберігаємо синтетичні дані
    synth_df = pd.DataFrame({
        'combined': synthetic_info['combined'],
        'trend': synthetic_info['trend'],
        'seasonal': synthetic_info['seasonal'],
        'noise': synthetic_info['noise']
    })
    
    synth_path = output_dir / 'synthetic_data.csv'
    synth_df.to_csv(synth_path, index=False)
    print(f"  Синтетичні дані збережено: {synth_path}")
    
    return synthetic_info, synth_props


def main():
    args = parse_arguments()
    config = vars(args)
    
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS TOOLKIT")
    print("Alpha-Beta Filtering + Advanced Analysis")
    print("="*60)
    
    # Завантаження даних
    try:
        if args.file:
            print(f"\nЗавантаження з файлу: {args.file}")
            df_raw = pd.read_csv(args.file)
        else:
            print(f"\nЗавантаження з URL: {args.url}")
            df_raw = dl.fetch_data(args.url)
        
        if df_raw is None:
            raise ValueError("Не вдалося завантажити дані")
        
        print(f"  Завантажено {len(df_raw)} рядків")
        
        # Підготовка даних
        print("\n[PREP] Обробка даних...")
        df_prepared = dh.prepare_timeseries(df_raw)
        
        # Директорія для виводу
        output_dir = Path(__file__).parent / args.output_dir
        output_dir.mkdir(exist_ok=True)
        
        # Виконання відповідно до режиму
        if args.mode == 'filtering':
            mode_filtering(df_raw, config, output_dir)
            
        elif args.mode == 'analysis':
            mode_analysis(df_prepared, config, output_dir)
            
        elif args.mode == 'synthetic':
            analysis_result = mode_analysis(df_prepared, config, output_dir)
            mode_synthetic(df_prepared, analysis_result, config, output_dir)
            
        elif args.mode == 'full':
            # Повний цикл: фільтрація → аналіз → синтетичні
            print("\n" + "="*60)
            print("ПОВНИЙ ЦИКЛ АНАЛІЗУ")
            print("="*60)
            
            # 1. Фільтрація
            df_filtered, metrics_result = mode_filtering(df_raw, config, output_dir)
            
            # 2. Аналіз
            analysis_result = mode_analysis(df_prepared, config, output_dir)
            
            # 3. Синтетичні
            synthetic_info, synth_props = mode_synthetic(
                df_prepared, analysis_result, config, output_dir
            )
            
            print("\n" + "="*60)
            print("ПОВНИЙ ЦИКЛ ЗАВЕРШЕНО")
            print("="*60)
        
        print(f"\n✓ Успішно завершено. Результати у {output_dir}")
        
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("ПОМИЛКА")
        print("="*60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()