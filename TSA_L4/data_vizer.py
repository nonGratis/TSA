import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from scipy.stats import chi2

COLOR_PRIMARY = '#1323e9'
COLOR_SECONDARY = '#ffaa3a'
COLOR_ACCENT = '#eb5f54'
COLOR_BLACK = '#000000'
COLOR_GRAY = '#cccccc'


def plot_data_preprocessing(
    df_raw: pd.DataFrame,
    df_resampled: pd.DataFrame,
    title: str = "Етап 0: Попередня обробка даних (Сирі виміри vs Регулярна сітка)",
    save_path: Optional[str] = None
) -> None:
    """
    Візуалізація сирих даних та результату ресемплінгу/імпутації.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Підготовка сирих даних для відображення
    if 'timestamp' in df_raw.columns:
        # Намагаємося розпарсити час, якщо він ще не є datetime
        if not pd.api.types.is_datetime64_any_dtype(df_raw['timestamp']):
            t_raw = pd.to_datetime(df_raw['timestamp'], dayfirst=True, errors='coerce')
        else:
            t_raw = df_raw['timestamp']
    else:
        t_raw = df_raw.index

    # Отримуємо значення, ігноруючи NaN
    if 'r_id' in df_raw.columns:
        val_raw = pd.to_numeric(df_raw['r_id'], errors='coerce')
    else:
        val_raw = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce')
        
    mask_raw_valid = ~np.isnan(val_raw) & ~np.isnat(t_raw) if hasattr(t_raw, 'dt') else ~np.isnan(val_raw)
    
    # 2. Малюємо СИРІ дані (дрібні точки)
    ax.scatter(t_raw[mask_raw_valid], val_raw[mask_raw_valid], 
               color=COLOR_GRAY, s=5, alpha=0.6, label='Сирі виміри (Raw Data)', zorder=1)
    
    # 3. Малюємо РЕСЕМПЛІНГ (Крива)
    t_res = df_resampled.index
    val_res = df_resampled['r_id_raw']
    
    ax.plot(t_res, val_res, '-', color=COLOR_PRIMARY, linewidth=1.5, alpha=0.8,
            label='Ресемплінг (1 година)', zorder=2)
            
    # 4. Підсвічуємо ІМПУТОВАНІ точки
    if 'is_imputed' in df_resampled.columns:
        mask_imp = df_resampled['is_imputed'].astype(bool)
        if mask_imp.any():
            ax.scatter(t_res[mask_imp], val_res[mask_imp], 
                       color=COLOR_SECONDARY, s=40, marker='x', linewidth=1.5, 
                       label='Імпутовані точки (Відновлено)', zorder=3)
            
            # 5. Метадані (Статистика)
            n_raw = mask_raw_valid.sum()
            n_grid = len(df_resampled)
            n_imp = mask_imp.sum()
            pct_imp = (n_imp / n_grid) * 100
            
            stats_text = (
                f"Статистика набору даних:\n"
                f"------------------------\n"
                f"Вхідні виміри: {n_raw}\n"
                f"Регулярна сітка: {n_grid} точок\n"
                f"Імпутовано: {n_imp} ({pct_imp:.1f}%)"
            )
            
            # Розміщуємо текст у "легенді" або окремому боксі
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLOR_GRAY)
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, fontfamily='monospace')

    # Форматування часу
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=0)

    ax.set_xlabel('Час (UTC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Значення лічильника', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_kalman_results(
    df: pd.DataFrame,
    k_steps_prediction: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    title: str = "Результати Kalman-фільтрації",
    save_path: Optional[str] = None
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    t = np.arange(len(df))

    ax.plot(t, df['r_id_raw'], 'o', color=COLOR_GRAY, markersize=3,
            alpha=0.5, label='Дані', zorder=1)

    # Імпутовані точки
    imputed_idx = df['is_imputed'].astype(bool).values
    if imputed_idx.any():
        ax.scatter(t[imputed_idx], df['r_id_raw'].values[imputed_idx],
                   color=COLOR_SECONDARY, s=80, marker='x', linewidths=2,
                   label='Імпутовано', zorder=3)

    # Аномалії
    anomaly_idx = df['is_anomaly'].astype(bool).values
    if anomaly_idx.any():
        ax.scatter(t[anomaly_idx], df['r_id_raw'].values[anomaly_idx],
                   color=COLOR_ACCENT, s=100, marker='D',
                   label='Аномалії', zorder=4)

    # Kalman-фільтр
    ax.plot(t, df['kf_x'], color=COLOR_PRIMARY, linewidth=2.5,
            label='Kalman-фільтр', zorder=2)

    # k-кроковий прогноз (+ CI якщо передано variance)
    if k_steps_prediction is not None:
        if isinstance(k_steps_prediction, tuple):
            preds, vars_pos = k_steps_prediction
            preds = np.asarray(preds, dtype=float)
            stds = np.sqrt(np.maximum(np.asarray(vars_pos, dtype=float), 0.0))
        else:
            preds = np.asarray(k_steps_prediction, dtype=float)
            stds = None

        k = len(preds)
        t_future = np.arange(len(df), len(df) + k)

        ax.plot(t_future, preds, '--', color=COLOR_PRIMARY,
                linewidth=2, label=f'Прогноз {k} кроків', alpha=0.8, zorder=2)
        ax.axvline(x=len(df) - 1, color=COLOR_GRAY, linestyle=':', linewidth=1.5, alpha=0.7)

        ax.relim()
        ax.autoscale_view()
        orig_ylim = ax.get_ylim()

        if stds is not None:
            ci = 1.96 * stds
            upper = preds + ci
            lower = preds - ci
            ax.fill_between(t_future, lower, upper, color=COLOR_PRIMARY, alpha=0.15,
                            label='95% CI прогнозу', zorder=1, clip_on=True)
            ax.set_ylim(orig_ylim)

    ax.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Накопичувальний лічильник', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_residuals_analysis(
    df: pd.DataFrame,
    title: str = "Аналіз залишків",
    save_path: Optional[str] = None
) -> None:
    has_alpha = 'alpha' in df.columns
    has_q = 'process_q' in df.columns
    has_nis = 'nis' in df.columns

    # Якщо є адаптивні параметри, малюємо 2 графіки
    if has_alpha or has_q or has_nis:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

    t = np.arange(len(df))
    residuals = np.asarray(df['residual'].values, dtype=float)

    # --- Графік 1: Залишки ---
    ax1.scatter(t, residuals, color=COLOR_PRIMARY, s=20, alpha=0.6, label='Залишки')
    ax1.axhline(y=0, color=COLOR_BLACK, linestyle='--', linewidth=1.5)

    window = 24
    if len(residuals) > window:
        # Ігноруємо NaN при розрахунку
        rolling_var = pd.Series(residuals).rolling(window=window, center=True).var()
        rolling_std = np.sqrt(rolling_var)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(t, rolling_std, color=COLOR_ACCENT, linewidth=2,
                      alpha=0.7, label=f'Ковзна σ (вікно={window})')
        ax1_twin.fill_between(t, -2*rolling_std, 2*rolling_std,
                              color=COLOR_ACCENT, alpha=0.1)
        ax1_twin.set_ylabel('Ковзна σ', fontsize=11, color=COLOR_ACCENT)
        ax1_twin.legend(fontsize=9, loc='upper right')

    ax1.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Залишок (Residual)', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Графік 2: Адаптивні параметри (Alpha або NIS) ---
    if has_alpha or has_q or has_nis:
        if has_alpha:
            alpha_vals = np.asarray(df['alpha'].values, dtype=float)
            ax2.plot(t, alpha_vals, color=COLOR_SECONDARY, linewidth=2, label='Alpha (Gain)')
            ax2.set_ylabel('Alpha', fontsize=12, fontweight='bold', color=COLOR_SECONDARY)
            ax2.set_ylim(0, 1.05)
            
            # Якщо є ще й Q, покажемо на осі Y2
            if has_q:
                q_vals = np.asarray(df['process_q'].values, dtype=float)
                ax2_twin = ax2.twinx()
                ax2_twin.plot(t, q_vals, color=COLOR_GRAY, linewidth=1, linestyle='--', alpha=0.5, label='Process Q')
                ax2_twin.set_yscale('log')
                ax2_twin.set_ylabel('Process Noise Q (log)', fontsize=10, color=COLOR_GRAY)
        
        elif has_nis:
            # Якщо Alpha стабільна, але є NIS - малюємо NIS
            nis_vals = np.asarray(df['nis'].values, dtype=float)
            ax2.plot(t, nis_vals, color=COLOR_PRIMARY, linewidth=1.5, label='NIS')
            
            upper = chi2.ppf(0.975, df=1)
            ax2.axhline(y=upper, color=COLOR_ACCENT, linestyle='--', label='95% межа')
            ax2.set_ylabel('NIS', fontsize=12, fontweight='bold')
            ax2.set_yscale('log')

        ax2.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
        ax2.set_title('Параметри адаптації фільтра', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_acf_histogram(
    residuals: np.ndarray,
    acf_values: np.ndarray,
    confidence_bound: float,
    title: str = "Діагностика залишків",
    save_path: Optional[str] = None
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Remove NaNs
    valid_res = residuals[~np.isnan(residuals)]

    # Гістограма
    ax1.hist(valid_res, bins=30, color=COLOR_PRIMARY, alpha=0.7,
             edgecolor=COLOR_BLACK, density=True)
    ax1.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=2)

    if len(valid_res) > 1:
        mu, sigma = np.mean(valid_res), np.std(valid_res)
        x = np.linspace(valid_res.min(), valid_res.max(), 100)
        normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax1.plot(x, normal_dist, color=COLOR_ACCENT, linewidth=2,
                 label=f'N({mu:.2f}, {sigma:.2f}²)')

    ax1.set_xlabel('Значення залишку', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Щільність', fontsize=11, fontweight='bold')
    ax1.set_title('Розподіл залишків', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # ACF
    lags = np.arange(len(acf_values))
    ax2.stem(lags, acf_values, linefmt=COLOR_PRIMARY, markerfmt='o', basefmt=COLOR_BLACK)

    ax2.axhline(y=confidence_bound, color=COLOR_ACCENT, linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'±{confidence_bound:.3f} (95%)')
    ax2.axhline(y=-confidence_bound, color=COLOR_ACCENT, linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color=COLOR_BLACK, linewidth=1)

    ax2.set_xlabel('Лаг', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ACF', fontsize=11, fontweight='bold')
    ax2.set_title('Автокореляційна функція (ACF)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_velocity_analysis(
    df: pd.DataFrame,
    title: str = "Аналіз швидкості змін",
    save_path: Optional[str] = None
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    t = np.arange(len(df))
    velocity = np.asarray(df['kf_v'].values, dtype=float)

    ax.plot(t, velocity, color=COLOR_PRIMARY, linewidth=2)
    ax.axhline(y=0, color=COLOR_BLACK, linestyle='--', linewidth=1.5)

    mean_v = float(np.mean(velocity))
    std_v = float(np.std(velocity))
    ax.axhline(y=mean_v, color=COLOR_ACCENT, linestyle=':', linewidth=2,
               alpha=0.7, label=f'Середня швидкість: {mean_v:.2f}')
    ax.fill_between(t, mean_v - std_v, mean_v + std_v,
                    color=COLOR_ACCENT, alpha=0.1,
                    label=f'±1σ ({std_v:.2f})')

    ax.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Швидкість (Δr_id/год)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()