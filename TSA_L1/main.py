import sys
import data_loader as dl
import data_handler as dh
import data_analyser as da
import data_vizer as dv

def main():
    if len(sys.argv) < 2:
        print("Використання: python main.py <URL>")
        sys.exit(1)

    url = sys.argv[1]

    raw_df = dl.fetch_data(url)
    
    if raw_df is None:
        print("Помилка: Не вдалося завантажити або обробити дані.")
        sys.exit(1)
    
    da.df_info(raw_df)
        
    clean_df = dh.prepare_timeseries(raw_df)
    
    print("Оберіть тип моделі тренду:")
    print("  1 - Поліноміальна")
    print("  2 - Логарифмічна")
    model_choice = input("Номер:").strip()
    
    if model_choice == '1':
        degree = int(input("Введіть ступінь полінома: "))
        X, y, y_trend, coeffs = da.fit_polynomial_trend(clean_df, degree)
        model_type = 'poly'
    elif model_choice == '2':
        X, y, y_trend, coeffs = da.fit_logarithmic_trend(clean_df)
        model_type = 'log'
    else:
        print("Помилка: Невірний вибір моделі.")
        sys.exit(1)
    
    residuals = da.calculate_residuals(y, y_trend)
    real_vel, model_vel = da.calculate_process_velocity(y, trend_func, X)
    
    print(f"Коєфіцієнти моделі: {coeffs}")
    
    mean, variance, std = da.calculate_statistics(residuals)    
    print(f"\n--- Статистика залишків ---")
    print(f"Математичне сподівання (M): {mean:.4f}")
    print(f"Дисперсія (D): {variance:.4f}")
    print(f"Стандартне відхилення (σ): {std:.4f}")
    
    dv.plot_comprehensive_report(clean_df.index, y, y_trend, residuals, degree, coeffs, real_vel, model_vel)

if __name__ == "__main__":
    main()