import sys
import data_loader as dl
import data_handler as dh
import data_analyser as da
import data_vizer as dv

def main():
    # Перевірка аргументів командного рядка
    if len(sys.argv) < 3:
        print("Використання: python main.py <URL> <СТУПІНЬ_ПОЛІНОМА>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        degree = int(sys.argv[2])
    except ValueError:
        print("Помилка: Ступінь має бути цілим числом")
        sys.exit(1)

    print(f"--- Запуск аналізу (Ступінь: {degree}) ---")

    raw_df = dl.fetch_data(url)
    
    if raw_df is None:
        print("Помилка: Не вдалося завантажити або обробити дані.")
        sys.exit(1)
        
    clean_df = dh.prepare_timeseries(raw_df)

    X, y, y_trend, coeffs, trend_func = da.fit_trend_model(clean_df, degree)
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