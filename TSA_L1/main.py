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
    
    print("\n\nТип моделі тренду:")
    print("    1 - Поліноміальна")
    print("    2 - Логарифмічна")
    model_choice = input("Номер:").strip()
    
    if model_choice == '1':
        degree = int(input("Степінь полінома: "))
        X, y, y_trend, coeffs = da.fit_polynomial_trend(clean_df, degree)
        model_type = 'poly'
        num_params = degree + 1
    elif model_choice == '2':
        X, y, y_trend, coeffs = da.fit_logarithmic_trend(clean_df)
        model_type = 'log'
        num_params = 2
    else:
        print("Помилка: Невірний вибір моделі.")
        sys.exit(1)
    print(f"\nКоєфіцієнти моделі (старший-молодший): {coeffs}")
    
    residuals = da.calculate_residuals(y, y_trend)
    _, _, resid_std = da.calculate_statistics(residuals)    
    
    r_squared = da.calculate_r_squared(y, y_trend)
    adj_r_squared = da.calculate_adjusted_r_squared(r_squared, len(y), num_params)
    print(f"\nКоефіцієнт детермінації R²: {r_squared:.4f}")
    print(f"Скоригований R²: {adj_r_squared:.4f}")
    
    f_stat, f_p_value = da.calculate_f_statistic(r_squared, num_params, len(y))
    print(f"Критерій Фішера F-statistic: {f_stat:.2f}, p-value: {f_p_value:.2e}")
    
    p_value = da.check_normality(residuals)
    print(f"\nТест нормальності (Шапіро-Вілка) залишків теоретичної моделі, p-value: {p_value:.4e}")
    
    print("\nТип розподілу генерації шуму:")
    print("    1 - Нормальний")
    print("    2 - Рівномірний")
    print("    3 - Експоненціальний")
    noise_choice = input("Номер:").strip()
    
    noise_map = {'1': 'normal', '2': 'uniform', '3': 'exponential'}
    distribution = noise_map.get(noise_choice, 'normal')
    
    y_synthetic = da.generate_synthetic_data(y_trend, resid_std, distribution)
    residuals_synthetic = da.calculate_residuals(y_trend, y_synthetic)
    residuals_synthetic_to_reality = da.calculate_residuals(y, y_synthetic)
    
    print(f"\n{'Компонента':<30} | {'M (μ)':<12} | {'D (σ²)':<12} | {'Std (σ)':<12}")
    print("-" * 75)

    datasets = {
        "Експерементальні дані": y,
        "Теоретична модель": y_trend,
        "Залишки теоретичної моделі": residuals,
        "Синтетична модель": y_synthetic,
        "Залишки синтетичної моделі": residuals_synthetic,
        "Залишки синт. моделі до даних": residuals_synthetic_to_reality
    }
    for name, data in datasets.items():
        m, v, s = da.calculate_statistics(data)
        print(f"{name:<30} | {m:<12.2f} | {v:<12.2f} | {s:<12.2f}")
    print("-" * 75)
        
    
    dv.plot_comprehensive_report(clean_df.index, y, y_trend, residuals, y_synthetic, residuals_synthetic_to_reality, model_type, coeffs)

if __name__ == "__main__":
    main()