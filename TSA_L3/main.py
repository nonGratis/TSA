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
from kalman import KalmanFilter


def load_data(args):
    """Завантаження даних з файлу або URL"""
    if args.file:
        print(f"Завантаження даних з файлу: {args.file}")
        df = pd.read_csv(args.file)
        print(f"  Завантажено {len(df)} рядків")
        return df
    elif args.url:
        print(f"Отримання даних з URL...")
        df = dl.fetch_data(args.url)
        if df is None:
            raise ValueError("Не вдалося отримати дані з URL")
        return df
    else:
        raise ValueError("Потрібно вказати --url або --file")


def build_config(args) -> dict:
    """Побудова конфігурації з аргументів командного рядка"""
    config = {
        'state_dim': args.state_dim,
        'dt': 1.0,  # Годинний інтервал
        'process_noise': args.process_noise,
        'measurement_noise': args.measurement_noise,
        'robust_estimation': args.robust,
        'adaptive': args.adaptive,
        'adaptive_window': args.window,
        'q_min': args.q_min,
        'q_max': args.q_max,
        'adapt_rate': args.adapt_rate,
        'imputed_update_mode': args.imputed_mode,
        'k_steps': args.k_steps
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Kalman-фільтрація кумулятивних часових рядів',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Джерело даних
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--file', type=str, help='Шлях до CSV файлу')
    data_group.add_argument('--url', type=str, help='URL Google Sheets')
    
    # Параметри Kalman-фільтра
    parser.add_argument('--state-dim', type=int, default=2, choices=[2, 3],
                       help='Розмірність стану: 2=[x,v], 3=[x,v,a] (default: 2)')
    parser.add_argument('--process-noise', type=float, default=None,
                       help='Процесний шум Q (якщо None - автооцінка)')
    parser.add_argument('--measurement-noise', type=float, default=None,
                       help='Шум вимірювання R (якщо None - автооцінка)')
    parser.add_argument('--robust', action='store_true', default=True,
                       help='Робастна оцінка шуму через MAD (default: True)')
    parser.add_argument('--no-robust', dest='robust', action='store_false',
                       help='Використовувати звичайну variance замість MAD')
    
    # Адаптивний Q
    parser.add_argument('--adaptive', action='store_true', default=True,
                       help='Використовувати адаптивний Q (default: True)')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false',
                       help='Вимкнути адаптивний Q')
    parser.add_argument('--window', type=int, default=24,
                       help='Розмір вікна для адаптивного Q (default: 24)')
    parser.add_argument('--q-min', type=float, default=1e-6,
                       help='Мінімальний Q (default: 1e-6)')
    parser.add_argument('--q-max', type=float, default=1e2,
                       help='Максимальний Q (default: 1e2)')
    parser.add_argument('--adapt-rate', type=float, default=1.2,
                       help='Швидкість адаптації Q (default: 1.2)')
    
    # Режим роботи з імпутованими даними
    parser.add_argument('--imputed-mode', type=str, default='skip',
                       choices=['skip', 'weighted', 'normal'],
                       help='Режим обробки імпутованих: skip (тільки predict), '
                            'weighted (update з великим R), normal (звичайний update)')
    
    # Прогнозування
    parser.add_argument('--k-steps', type=int, default=24,
                       help='Кількість кроків для прогнозу (default: 24)')
    
    args = parser.parse_args()
    
    try:
        # Завантаження даних
        df_raw = load_data(args)
        
        # Побудова конфігурації
        config = build_config(args)
        
        print("КОНФІГУРАЦІЯ ФІЛЬТРА")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        
        # Запуск пайплайну
        df_result = pl.run_pipeline(df_raw, config)
        
        # Обчислення метрик (тільки на валідних вимірах, не на імпутованих)
        valid_mask = np.asarray(df_result['valid_measurement'].values, dtype=bool)
        
        print("\n" + "="*60)
        print(f"Валідація на {valid_mask.sum()} валідних вимірах (виключено {(~valid_mask).sum()} імпутованих)")
        
        metrics_result = mt.evaluate_filter_performance(
            y_true=np.asarray(df_result['r_id_raw'].values[valid_mask], dtype=float),
            y_filtered=np.asarray(df_result['kf_x'].values[valid_mask], dtype=float),
            residuals=np.asarray(df_result['residual'].values[valid_mask], dtype=float),
            verbose=True
        )
        
        # Перевірка на білий шум (тільки валідні залишки)
        valid_residuals = np.asarray(df_result['residual'].values[valid_mask], dtype=float)
        
        whiteness = mt.check_residuals_whiteness(
            valid_residuals,
            nlags=min(40, len(valid_residuals) // 2)
        )
        
        if whiteness['is_white_noise']:
            print("✓ Залишки є білим шумом (некорельовані)")
        else:
            print(f"✗ Залишки мають кореляцію на лагах: {whiteness['significant_lags'][:5]}...")
        
        
        print(f"\nПрогноз {args.k_steps} кроків вперед...")
        
        # Використаємо оцінку залишків для ініціалізації P (щоб CI були більш інформативні)
        residual_std = metrics_result.get('residual_std', 1.0)
        init_P = np.eye(config['state_dim']) * max(residual_std**2, 1e-6)
        
        # Створюємо новий фільтр з фінальним станом та ініціалізованим P
        init_state = np.array([df_result['kf_x'].iloc[-1], df_result['kf_v'].iloc[-1]])
        final_kf = KalmanFilter(
            dt=config['dt'],
            state_dim=config['state_dim'],
            process_noise_q=metrics_result.get('residual_std', 1.0) ** 2,
            measurement_noise_r=config['measurement_noise'],
            init_state=init_state,
            init_P=init_P
        )
        
        k_predictions, k_vars = final_kf.predict_k_steps(args.k_steps)
        k_stds = np.sqrt(np.maximum(k_vars, 0.0))
        
        print(f"  Прогноз на наступні {args.k_steps} годин:")
        print(f"    Поточне значення: {df_result['kf_x'].iloc[-1]:.2f}")
        print(f"    Прогноз через {args.k_steps}год: {k_predictions[-1]:.2f} ± {1.96*k_stds[-1]:.2f} (95% CI)")
        print(f"    Приріст (точково): {k_predictions[-1] - df_result['kf_x'].iloc[-1]:.2f}")
        
        
        # Створюємо директорію images якщо її немає
        images_dir = Path(__file__).parent / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # 1. Головний графік з результатами
        dv.plot_kalman_results(df_result, (k_predictions, k_vars),
                            title="Kalman-фільтрація кумулятивного ряду",
                            save_path=str(images_dir / '01_kalman_results.svg'))

        
        # 2. Аналіз залишків
        dv.plot_residuals_analysis(df_result,
                                  title="Аналіз залишків та адаптації Q",
                                  save_path=str(images_dir / '02_residuals_analysis.svg'))
        
        # 3. ACF та гістограма (тільки валідні залишки)
        dv.plot_acf_histogram(
            valid_residuals,
            whiteness['acf_values'],
            whiteness['confidence_bound'],
            title="Діагностика залишків (тільки валідні виміри)",
            save_path=str(images_dir / '03_acf_histogram.svg')
        )
        
        # 4. Аналіз швидкості
        dv.plot_velocity_analysis(df_result,
                                title="Динаміка швидкості змін",
                                save_path=str(images_dir / '04_velocity_analysis.svg'))
        
        print(f"  Візуалізації збережено в: {images_dir.absolute()}")
        
    except Exception as e:
        print(f"\nПОМИЛКА: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

