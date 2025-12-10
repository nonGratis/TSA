import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import data_loader as dl
import data_vizer as dv
import pipeline as pl
import metrics as mt
from kalman import AlphaBetaFilter

def main():
    parser = argparse.ArgumentParser(description='Alpha-Beta фільтрація (Refactored)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Шлях до CSV файлу')
    group.add_argument('--url', type=str, help='URL до Google Sheet')
    
    parser.add_argument('--state-dim', type=int, default=2, choices=[2, 3], help="Розмірність стану (2=CV, 3=CA)")
    parser.add_argument('--process-noise', type=float, default=None, help="Process noise Q (базове значення)")
    parser.add_argument('--measurement-noise', type=float, default=None, help="Measurement noise R (фіксоване)")
    
    parser.add_argument('--adaptive', action='store_true', default=True, help="Увімкнути NIS адаптацію")
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
    
    parser.add_argument('--k-steps', type=int, default=12, help="Кількість кроків прогнозу")
    parser.add_argument('--imputed-mode', dest='imputed_update_mode', type=str, default='skip',
                      choices=['skip', 'weighted'], help="Стратегія обробки імпутованих точок")
    

    args = parser.parse_args()
    

    if args.file:
        df_raw = pd.read_csv(args.file)
    else:
        df_raw = dl.fetch_data(args.url)
    
    if df_raw is None:
        raise ValueError("Не вдалося завантажити дані.")
        
    config = vars(args)
    config['dt'] = 1.0
    
    df_res = pl.run_pipeline(df_raw, config)
    
    mask_valid = ~np.isnan(df_res['residual'])
    metrics = mt.evaluate_filter_performance(
        np.asarray(df_res['r_id_raw'][mask_valid].values, dtype=float), 
        np.asarray(df_res['kf_x'][mask_valid].values, dtype=float), 
        np.asarray(df_res['residual'][mask_valid].values, dtype=float)
    )
    
    print(f"\nПрогноз {args.k_steps} кроків...")
    
    last_x = float(df_res['kf_x'].iloc[-1])
    last_v = float(df_res['kf_v'].iloc[-1])
    
    init_state = [last_x, last_v]
    if args.state_dim == 3 and 'kf_a' in df_res.columns:
        init_state.append(float(df_res['kf_a'].iloc[-1]))
        
    est_std = metrics.get('residual_std', 1.0)
    pred_r = float(args.measurement_noise) if args.measurement_noise else float(est_std**2)
    if pred_r <= 0: pred_r = 1.0

    pred_filter = AlphaBetaFilter(
        dt=1.0, 
        state_dim=args.state_dim,
        init_state=np.array(init_state),
        measurement_noise_r=pred_r
    )
    
    preds, vars_pred = pred_filter.predict_k_steps(args.k_steps)
    stds = np.sqrt(vars_pred)
    
    print(f"  Next (+1h): {preds[0]:.2f}")
    print(f"  End (+{args.k_steps}h): {preds[-1]:.2f} ± {1.96*stds[-1]:.2f} (95% CI)")
    
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True)
    
    dv.plot_data_preprocessing(
        df_raw, 
        df_res, 
        title="Попередня обробка даних та імпутація",
        save_path=str(images_dir / '00_data_prep.svg')
    )

    dv.plot_kalman_results(df_res, (preds, vars_pred), 
                        title="Результати фільтра Калмана (Alpha/NIS)",
                        save_path=str(images_dir / '01_kf_results.svg'))
    
    dv.plot_residuals_analysis(df_res, title="Аналіз залишків та адаптації",
                            save_path=str(images_dir / '02_kf_diagnostics.svg'))                    
if __name__ == "__main__":
    main()