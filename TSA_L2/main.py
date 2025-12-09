import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from data_loader import fetch_data
from data_handler import prepare_timeseries
from model_synthesizer import PolynomialModel
from logistic_model import LogisticRND
from model_comparator import ModelComparator
from data_vizer import DataVisualizer

def load_data(url: Optional[str] = None, filepath: Optional[str] = None) -> pd.DataFrame:
    if filepath:
        print(f"Завантаження даних з файлу: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  Завантажено {len(df)} рядків")
        return df
    elif url:
        print(f"Отримання даних з URL...")
        df = fetch_data(url)
        if df is None:
            raise ValueError("Не вдалося отримати дані з URL")
        return df
    else:
        raise ValueError("Потрібно вказати --url або --file")


def run_polynomial_model(y: np.ndarray) -> dict:
    """
    Train polynomial model (degree=2) and return results
    
    Args:
        y: Time series data
        
    Returns:
        Dictionary with model, predictions, metrics
    """
    print("\nПОЛІНОМІАЛЬНА МОДЕЛЬ (ст. 2)")
    
    model = PolynomialModel(y)
    model.fit()
    
    x_train = np.arange(len(y))
    y_pred = model.predict(x_train)
    
    print(f"\nРівняння: {model.get_equation_string()}")
    print(f"Параметри: {model.get_params()}")
    
    return {
        'model': model,
        'predictions': y_pred,
        'X': x_train
    }


def export_timeseries(df: pd.DataFrame, file_path: str) -> None:
    """
    Експортує оброблений Time Series у CSV згідно зі специфікацією.
    
    Format: r_id, timestamp, imputation
    """
    # 1. Створюємо копію, щоб не мутувати оригінальний DF у пам'яті (Pure Function approach)
    out_df = df.copy()
    
    # 2. Витягуємо timestamp з індексу в колонку
    out_df = out_df.reset_index()
    
    # 3. Перейменовуємо колонки для відповідності вимогам (imputed -> imputation)
    # Припускаємо, що ім'я індексу вже 'timestamp', але якщо ні - перейменовуємо примусово
    rename_map = {
        out_df.columns[0]: 'timestamp',  # перша колонка після reset_index - це час
        'imputed': 'imputation'
    }
    out_df = out_df.rename(columns=rename_map)
    
    # 4. Жорсткий відбір та впорядкування колонок (Schema Enforcement)
    target_columns = ['r_id', 'timestamp', 'imputation']
    
    # Перевірка наявності всіх колонок перед збереженням
    if not set(target_columns).issubset(out_df.columns):
        raise ValueError(f"Missing columns for export. Expected {target_columns}, got {out_df.columns.tolist()}")

    # 5. Збереження
    # float_format='%.3f' використовуємо, щоб r_id не зберігався як 100.00000001
    out_df[target_columns].to_csv(file_path, index=False, float_format='%.6g')
    print(f"Часова послідовність збережена в: {file_path}")

def run_logistic_model(y: np.ndarray, epochs: int = 5000) -> dict:
    """
    Train logistic model with gradient descent and return results
    
    Args:
        y: Time series data
        epochs: Number of training epochs
        
    Returns:
        Dictionary with model, predictions, metrics
    """
    print("\nЛОГІСТИЧНА МОДЕЛЬ")
    
    X_train = np.arange(len(y))
    
    model = LogisticRND(learning_rate=0.01, epochs=epochs, optimizer='adam')
    print(f"\nНавчання: {epochs} епох")
    model.fit(X_train, y, verbose=True)
    
    y_pred = model.predict(X_train)
    
    print(f"\nКінцеве рівняння: {model.get_equation_string()}")
    print(f"Параметри: {model.get_params()}")
    
    return {
        'model': model,
        'predictions': y_pred,
        'X': X_train
    }


def compare_models(y_true: np.ndarray, poly_result: dict, logistic_result: dict) -> dict:
    """
    Compare polynomial and logistic models
    
    Args:
        y_true: Actual values
        poly_result: Polynomial model results
        logistic_result: Logistic model results
        
    Returns:
        Comparison dictionary
    """
    print("\nПОРІВНЯННЯ МОДЕЛЕЙ")
    
    comparator = ModelComparator()
    
    # Evaluate polynomial (3 parameters: a, b, c for degree-2)
    poly_metrics = comparator.evaluate_model(
        'Polynomial',
        y_true,
        poly_result['predictions'],
        n_params=3,
        params=poly_result['model'].get_params()
    )
    
    # Evaluate logistic (3 parameters: L, k, t0)
    logistic_metrics = comparator.evaluate_model(
        'Logistic',
        y_true,
        logistic_result['predictions'],
        n_params=3,
        params=logistic_result['model'].get_params()
    )
    
    # Compare
    comparison = comparator.compare(poly_metrics, logistic_metrics)
    comparator.print_summary(comparison)
    
    return comparison


def generate_visualizations(y: np.ndarray, imputed: np.ndarray, 
                           poly_result: dict, logistic_result: dict,
                           comparison: dict, output_dir: str = './plots') -> None:
    """
    Generate all visualization plots
    
    Args:
        y: Original data
        imputed: Boolean array of imputed points
        poly_result: Polynomial model results
        logistic_result: Logistic model results
        comparison: Comparison dictionary
        output_dir: Directory to save plots
    """
    print("\nФОРМУВАННЯ ВІЗУАЛІЗАЦІЙ")
    
    viz = DataVisualizer(output_dir)
    t = np.arange(len(y))
    
    # 1. Raw data
    print("\n[1/5] Побудова: сирі дані...")
    viz.plot_raw_data(t, y, imputed)
    
    # 2. Polynomial fit
    print("[2/5] Побудова: поліноміальна апроксимація...")
    viz.plot_polynomial_fit(
        t, y, poly_result['predictions'],
        poly_result['model'].get_equation_string(),
        comparison['polynomial']
    )
    
    # 3. Logistic fit
    print("[3/5] Побудова: логістична апроксимація...")
    viz.plot_logistic_fit(
        t, y, logistic_result['predictions'],
        logistic_result['model'].get_equation_string(),
        comparison['logistic'],
        logistic_result['model'].get_params()
    )
    
    # 4. Extrapolation
    print("[4/5] Побудова: екстраполяційне порівняння...")
    n_future = int(len(y) * 1.5)
    t_future = np.arange(n_future)
    y_poly_future = poly_result['model'].predict(t_future)
    y_logistic_future = logistic_result['model'].predict(t_future)
    
    viz.plot_extrapolation(t, y, t_future, y_poly_future, y_logistic_future)
    
    # Print extrapolation summary
    train_end = len(y) - 1
    extrap_end = len(t_future) - 1
    print(f"\nДіапазон навчання:     [0, {train_end}] год")
    print(f"Діапазон екстраполяції: [{train_end}, {extrap_end}] год")
    print(f"\nКінцеві значення:")
    print(f"  Поліном:  {y_poly_future[-1]:.0f} (зростає)")
    print(f"  Логістична:    {y_logistic_future[-1]:.0f}")
    
    # 5. Residuals histogram
    print("[5/5] Побудова: гістограма залишків...")
    residuals_poly = y - poly_result['predictions']
    residuals_logistic = y - logistic_result['predictions']
    viz.plot_residuals_histogram(residuals_poly, residuals_logistic)
    
    print(f"\nВсі графіки збережено в: {Path(output_dir).absolute()}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--file',
        type=str,
        help='Path to CSV file with time series data'
    )
    data_group.add_argument(
        '--url',
        type=str,
        help='Google Sheets URL to fetch data from'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='Number of training epochs for logistic model (default: 5000)'
    )
    
    args = parser.parse_args()
    
    try:
        df_raw = load_data(url=args.url, filepath=args.file)
        
        print("\nPreparing time series...")
        df_prepared = prepare_timeseries(df_raw)
        export_timeseries(df_prepared, 'cleaned_data.csv')
        y = np.asarray(df_prepared['r_id'].values, dtype=np.float64)
        imputed = np.asarray(df_prepared['imputed'].values, dtype=bool)
        
        print(f"  Time series length: {len(y)}")
        print(f"  Imputed points: {int(imputed.sum())}")
        print(f"  Range: [{float(y.min()):.0f}, {float(y.max()):.0f}]")
        
        poly_result = run_polynomial_model(y)
        logistic_result = run_logistic_model(y, args.epochs)

        comparison = compare_models(y, poly_result, logistic_result)
        
        generate_visualizations(y, imputed, poly_result, logistic_result, 
                               comparison, 'images')
                
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
