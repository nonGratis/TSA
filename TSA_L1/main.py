import sys
import argparse
import numpy as np
import data_loader as dl
import data_handler as dh
import data_analyser as da
import data_vizer as dv
from model_synthesizer import ModelSynthesizer

def main():
    parser = argparse.ArgumentParser(description='Аналіз часових рядів')
    parser.add_argument('url', help='URL Google Sheets')
    parser.add_argument('--seed', nargs='?', const=2330, type=int, default=None,
                        help='seed для встановлення зрізу випадкової генерації (default: 2330)')
    parser.add_argument('--model', choices=['poly', 'log'], help='Тип моделі')
    parser.add_argument('--degree', type=int, help='Степінь полінома')
    parser.add_argument('--noise', choices=['normal', 'uniform'], help='Тип розподілу шуму')
    
    args = parser.parse_args()

    raw_df = dl.fetch_data(args.url)
    if raw_df is None:
        print("Помилка: Не вдалося завантажити або обробити дані.")
        sys.exit(1)
    
    da.df_info(raw_df)
    clean_df = dh.prepare_timeseries(raw_df)
    da.set_random_seed(args.seed)
    
    y = np.array(clean_df['r_id'].values, dtype=float)
    synthesizer = ModelSynthesizer(y)
    
    # moдель
    if args.model: # Визначення моделі: мануально або автоматично
        model_type = args.model
        degree = args.degree if args.model == 'poly' else None
        y_trend, coeffs, model_type = synthesizer.build_trend(model_type, degree)
        distribution = args.noise if args.noise else 'normal'
        print(f"\n[Мануальний режим] Модель: {model_type}" + (f", degree={degree}" if degree else "") + f", шум: {distribution}")
    else:
        model_info = synthesizer.synthesize()
        y_trend, coeffs, model_type = synthesizer.build_trend(
            model_info['model_type'], 
            model_info['degree'], 
            model_info['coeffs']
        )
        distribution = args.noise if args.noise else model_info['recommended_distribution']
        if not args.noise:
            print(f"Визначений розподіл шуму: {distribution}")
    
    residuals = da.calculate_residuals(y, y_trend)
    y_synthetic = synthesizer.generate_synthetic_data(y_trend, residuals, distribution)
    residuals_synthetic = da.calculate_residuals(y_synthetic, y_trend)
    
    # звіт
    da.print_statistics_report(y, y_trend, y_synthetic, residuals, residuals_synthetic, coeffs, distribution)
    
    # віз
    dv.plot_report(clean_df.index, y, y_trend, residuals, y_synthetic, residuals_synthetic, model_type, coeffs)

if __name__ == "__main__":
    main()