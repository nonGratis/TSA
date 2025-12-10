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
    parser = argparse.ArgumentParser(description='Alpha-Beta фільтрація')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='CSV файл')
    group.add_argument('--url', type=str, help='URL')
    
    parser.add_argument('--state-dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--process-noise', type=float, default=None, help="Впливає на розрахунок Alpha")
    parser.add_argument('--measurement-noise', type=float, default=None, help="Впливає на розрахунок Alpha")
    
    parser.add_argument('--adaptive', action='store_true', default=True)
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
    
    parser.add_argument('--k-steps', type=int, default=12)
    parser.add_argument('--imputed-mode', dest='imputed_update_mode', type=str, default='skip')
    
    parser.add_argument('--robust', action='store_true', default=True)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--q-min', type=float, default=0)
    parser.add_argument('--q-max', type=float, default=0)
    parser.add_argument('--adapt-rate', type=float, default=0)

    args = parser.parse_args()
    
    try:
        if args.file:
            df = pd.read_csv(args.file)
        else:
            df = dl.fetch_data(args.url)
            
        config = vars(args)
        config['dt'] = 1.0
        
        # Pipeline
        df_res = pl.run_pipeline(df, config)
        
        # Metrics
        mask = df_res['valid_measurement']
        metrics = mt.evaluate_filter_performance(
            df_res['r_id_raw'][mask], 
            df_res['kf_x'][mask], 
            df_res['residual'][mask]
        )
        
        # Prediction
        print(f"\nПрогноз {args.k_steps} кроків...")
        
        last_alpha = df_res['q_value'].iloc[-1]
        last_x = df_res['kf_x'].iloc[-1]
        last_v = df_res['kf_v'].iloc[-1]
        
        init_state = [last_x, last_v]
        if args.state_dim == 3 and 'kf_a' in df_res.columns:
            init_state.append(df_res['kf_a'].iloc[-1])
            
        # Визначаємо R для прогнозу: аргумент CLI -> реальна статистика залишків -> дефолт
        if args.measurement_noise:
            pred_r = args.measurement_noise
        else:
            # R ~ Var(residuals) = std^2
            est_std = metrics.get('residual_std', 1.0)
            pred_r = est_std**2 if est_std > 1e-9 else 1.0

        pred_filter = AlphaBetaFilter(
            dt=1.0, 
            state_dim=args.state_dim,
            init_state=np.array(init_state),
            alpha=last_alpha,
            measurement_noise_r=pred_r
        )
        
        preds, vars_pred = pred_filter.predict_k_steps(args.k_steps)
        stds = np.sqrt(vars_pred)
        
        print(f"  Next: {preds[0]:.2f}, +{args.k_steps}h: {preds[-1]:.2f} ± {1.96*stds[-1]:.2f}")
        
        # Plotting
        images_dir = Path(__file__).parent / 'images'
        images_dir.mkdir(exist_ok=True)
        
        dv.plot_kalman_results(df_res, (preds, vars_pred), 
                             title="Alpha-Beta фільтрація",
                             save_path=str(images_dir / '01_ab_results.svg'))
        
        dv.plot_residuals_analysis(df_res, title="Residuals & Alpha",
                                 save_path=str(images_dir / '02_ab_residuals.svg'))
                                 
        print(f"  Done. Images in {images_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()