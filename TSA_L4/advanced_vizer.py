import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional, Dict
from scipy.stats import norm

COLOR_PRIMARY = '#1323e9'
COLOR_SECONDARY = '#ffaa3a'
COLOR_ACCENT = '#eb5f54'
COLOR_BLACK = '#000000'
COLOR_GRAY = '#cccccc'
COLOR_GREEN = '#2ecc71'
COLOR_PURPLE = '#9b59b6'


def plot_decomposition(decomposition: Dict, 
                      title: str = "Декомпозиція часового ряду (STL)",
                      save_path: Optional[str] = None) -> None:
    """
    Візуалізація результатів STL декомпозиції.
    
    Args:
        decomposition: Словник з компонентами (observed, trend, seasonal, resid)
        title: Заголовок графіку
        save_path: Шлях для збереження
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    observed = decomposition['observed']
    trend = decomposition['trend']
    seasonal = decomposition['seasonal']
    resid = decomposition['resid']
    
    # Observed
    axes[0].plot(observed.index, observed.values, color=COLOR_PRIMARY, linewidth=1.5)
    axes[0].set_ylabel('Observed', fontsize=11, fontweight='bold')
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(trend.index, trend.values, color=COLOR_ACCENT, linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(seasonal.index, seasonal.values, color=COLOR_SECONDARY, linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].scatter(resid.index, resid.values, color=COLOR_GRAY, s=10, alpha=0.6)
    axes[3].axhline(y=0, color=COLOR_BLACK, linestyle='--', linewidth=1)
    axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Час', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    # Форматування часу
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_stationarity_tests(data: pd.Series, 
                           stationarity_result: Dict,
                           save_path: Optional[str] = None, window_size=24) -> None:
    """
    Візуалізація результатів тестів стаціонарності.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Часовий ряд з ковзним середнім та σ
    ax = axes[0, 0]
    
    window = min(window_size, len(data) // 10)
    rolling_mean = data.rolling(window=window, center=True).mean()
    rolling_std = data.rolling(window=window, center=True).std()
    
    ax.plot(data.index, data.values, color=COLOR_GRAY, alpha=0.5, 
            linewidth=1, label='Дані')
    ax.plot(rolling_mean.index, rolling_mean.values, color=COLOR_PRIMARY, 
            linewidth=2, label=f'Ковзне μ (вікно={window})')
    ax.fill_between(data.index, 
                    rolling_mean - rolling_std, 
                    rolling_mean + rolling_std,
                    color=COLOR_PRIMARY, alpha=0.2, label='±1σ')
    
    ax.set_ylabel('Значення', fontsize=10, fontweight='bold')
    ax.set_title('Ковзні статистики', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. ADF та KPSS результати
    ax = axes[0, 1]
    ax.axis('off')
    
    adf = stationarity_result['adf']
    kpss = stationarity_result['kpss']
    
    text_content = "ТЕСТИ СТАЦІОНАРНОСТІ\n" + "="*35 + "\n\n"
    
    text_content += "ADF Test (H0: не стаціонарний):\n"
    text_content += f"  Test Statistic: {adf['test_statistic']:.4f}\n"
    text_content += f"  P-value: {adf['p_value']:.4f}\n"
    text_content += f"  → {'СТАЦІОНАРНИЙ' if adf['is_stationary'] else 'НЕ стаціонарний'}\n\n"
    
    if 'error' not in kpss:
        text_content += "KPSS Test (H0: стаціонарний):\n"
        text_content += f"  Test Statistic: {kpss['test_statistic']:.4f}\n"
        text_content += f"  P-value: {kpss['p_value']:.4f}\n"
        text_content += f"  → {'СТАЦІОНАРНИЙ' if kpss['is_stationary'] else 'НЕ стаціонарний'}\n\n"
    
    text_content += f"Висновок: {stationarity_result['conclusion'].upper()}"
    
    props = dict(boxstyle='round', facecolor='white', edgecolor=COLOR_GRAY)
    ax.text(0.1, 0.5, text_content, transform=ax.transAxes, 
            fontsize=10, verticalalignment='center',
            bbox=props, fontfamily='monospace')
    
    # 3. Розподіл значень
    ax = axes[1, 0]
    clean_data = data.dropna().values
    
    ax.hist(clean_data, bins=50, color=COLOR_PRIMARY, alpha=0.7, 
            edgecolor=COLOR_BLACK, density=True)
    
    # Накладаємо нормальний розподіл
    mu, sigma = np.mean(clean_data), np.std(clean_data)
    x = np.linspace(clean_data.min(), clean_data.max(), 100)
    ax.plot(x, norm.pdf(x, mu, sigma), color=COLOR_ACCENT, 
            linewidth=2, label=f'N({mu:.1f}, {sigma:.1f}²)')
    
    ax.set_xlabel('Значення', fontsize=10, fontweight='bold')
    ax.set_ylabel('Щільність', fontsize=10, fontweight='bold')
    ax.set_title('Розподіл значень', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Розподіл перших різниць
    ax = axes[1, 1]
    diffs = np.diff(clean_data)
    
    ax.hist(diffs, bins=50, color=COLOR_SECONDARY, alpha=0.7, 
            edgecolor=COLOR_BLACK, density=True)
    
    mu_diff, sigma_diff = np.mean(diffs), np.std(diffs)
    x = np.linspace(diffs.min(), diffs.max(), 100)
    ax.plot(x, norm.pdf(x, mu_diff, sigma_diff), color=COLOR_ACCENT, 
            linewidth=2, label=f'N({mu_diff:.2f}, {sigma_diff:.2f}²)')
    
    ax.set_xlabel('Δ Значення', fontsize=10, fontweight='bold')
    ax.set_ylabel('Щільність', fontsize=10, fontweight='bold')
    ax.set_title('Розподіл перших різниць', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Аналіз стаціонарності', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_hurst_and_acf(data: pd.Series, 
                      properties: Dict,
                      save_path: Optional[str] = None) -> None:
    """
    Візуалізація Hurst exponent та ACF/PACF.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    hurst = properties['hurst']
    autocorr = properties['autocorrelation']
    
    # 1. Інформація про Hurst
    ax = axes[0, 0]
    ax.axis('off')
    
    if hurst['hurst'] is not None:
        h_val = hurst['hurst']
        
        text_content = "ФРАКТАЛЬНИЙ АНАЛІЗ\n" + "="*30 + "\n\n"
        text_content += f"Hurst Exponent: {h_val:.4f}\n\n"
        text_content += f"Інтерпретація:\n{hurst['interpretation'].upper()}\n\n"
        text_content += f"{hurst['description']}\n\n"
        
        if h_val < 0.45:
            text_content += "Процес схильний до\nповернення до середнього"
            color = COLOR_ACCENT
        elif h_val > 0.55:
            text_content += "Процес має пам'ять\nта тренд продовжується"
            color = COLOR_GREEN
        else:
            text_content += "Процес випадковий\n(Brownian motion)"
            color = COLOR_GRAY
        
        props = dict(boxstyle='round', facecolor=color, alpha=0.2, edgecolor=color)
    else:
        text_content = f"Помилка обчислення Hurst:\n{hurst.get('error', 'Unknown')}"
        props = dict(boxstyle='round', facecolor='white', edgecolor=COLOR_GRAY)
    
    ax.text(0.5, 0.5, text_content, transform=ax.transAxes, 
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=props, fontfamily='monospace')
    
    # 2. Візуалізація Hurst через діаграму
    ax = axes[0, 1]
    
    if hurst['hurst'] is not None:
        h_val = hurst['hurst']
        
        # Шкала 0-1
        ax.barh([0], [1], color=COLOR_GRAY, alpha=0.2)
        ax.barh([0], [h_val], color=COLOR_PRIMARY, alpha=0.7)
        
        ax.axvline(x=0.5, color=COLOR_BLACK, linestyle='--', linewidth=2, alpha=0.5)
        ax.text(0.5, 0.5, 'H=0.5\n(Random)', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Hurst Exponent', fontsize=10, fontweight='bold')
        ax.set_yticks([])
        ax.set_title(f'H = {h_val:.3f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'Не вдалося обчислити', 
                ha='center', va='center', fontsize=11)
        ax.axis('off')
    
    # 3. ACF
    ax = axes[1, 0]
    
    acf_vals = autocorr['acf']
    lags = np.arange(len(acf_vals))
    conf_bound = autocorr['confidence_bound']
    
    ax.stem(lags, acf_vals, linefmt=COLOR_PRIMARY, 
            markerfmt='o', basefmt=COLOR_BLACK, label='ACF')
    ax.axhline(y=conf_bound, color=COLOR_ACCENT, linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'95% межа (±{conf_bound:.3f})')
    ax.axhline(y=-conf_bound, color=COLOR_ACCENT, linestyle='--', 
               linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color=COLOR_BLACK, linewidth=1)
    
    ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
    ax.set_ylabel('ACF', fontsize=10, fontweight='bold')
    ax.set_title('Автокореляційна функція (ACF)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    
    # 4. PACF
    ax = axes[1, 1]
    
    pacf_vals = autocorr['pacf']
    
    ax.stem(lags, pacf_vals, linefmt=COLOR_SECONDARY, 
            markerfmt='o', basefmt=COLOR_BLACK, label='PACF')
    ax.axhline(y=conf_bound, color=COLOR_ACCENT, linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'95% межа')
    ax.axhline(y=-conf_bound, color=COLOR_ACCENT, linestyle='--', 
               linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color=COLOR_BLACK, linewidth=1)
    
    ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
    ax.set_ylabel('PACF', fontsize=10, fontweight='bold')
    ax.set_title('Часткова автокореляційна функція (PACF)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    
    plt.suptitle('Фрактальний та Автокореляційний аналіз', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_clustering_results(data: pd.Series, 
                           clustering_result: Dict,
                           save_path: Optional[str] = None) -> None:
    """
    Візуалізація результатів кластеризації.
    """
    labels = clustering_result['labels']
    window_size = clustering_result['window_size']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 1. Часовий ряд з позначенням кластерів
    ax = axes[0]
    
    clean_data = data.dropna()
    
    # Малюємо весь ряд
    ax.plot(clean_data.index, clean_data.values, color=COLOR_GRAY, 
            linewidth=0.5, alpha=0.3, label='Дані')
    
    # Позначаємо кластери (центри вікон)
    unique_labels = np.unique(labels[labels != -1])
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        indices = np.where(mask)[0] + window_size // 2  # Центр вікна
        
        if len(indices) > 0:
            ax.scatter(clean_data.index[indices], clean_data.values[indices],
                      color=colors[i], s=30, alpha=0.7, 
                      label=f'Кластер {label}')
    
    # Позначаємо noise для DBSCAN
    if -1 in labels:
        mask = labels == -1
        indices = np.where(mask)[0] + window_size // 2
        ax.scatter(clean_data.index[indices], clean_data.values[indices],
                  color=COLOR_BLACK, s=20, alpha=0.5, marker='x',
                  label='Викиди')
    
    ax.set_ylabel('Значення', fontsize=11, fontweight='bold')
    ax.set_title('Кластеризація часового ряду', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=3)
    ax.grid(True, alpha=0.3)
    
    # 2. Розподіл кластерів
    ax = axes[1]
    
    cluster_stats = clustering_result['cluster_stats']
    cluster_labels = [f'Кластер {k}' for k in cluster_stats.keys()]
    cluster_sizes = [v['size'] for v in cluster_stats.values()]
    
    bars = ax.bar(cluster_labels, cluster_sizes, color=colors[:len(cluster_labels)], 
                   alpha=0.7, edgecolor=COLOR_BLACK)
    
    # Додаємо відсотки на стовпці
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        pct = size / sum(cluster_sizes) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Кількість вікон', fontsize=11, fontweight='bold')
    ax.set_title('Розподіл по кластерам', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_synthetic_vs_real(real_data: pd.Series, 
                          synthetic_data: np.ndarray,
                          save_path: Optional[str] = None) -> None:
    """
    Порівняння реальних та синтетичних даних.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    real_vals = real_data.dropna().values
    synth_vals = synthetic_data
    
    # Обрізаємо до однакової довжини
    min_len = min(len(real_vals), len(synth_vals))
    real_vals = real_vals[:min_len]
    synth_vals = synth_vals[:min_len]
    
    # 1. Часові ряди
    ax = axes[0, 0]
    ax.plot(real_vals, color=COLOR_PRIMARY, linewidth=1, alpha=0.7, label='Реальні')
    ax.set_ylabel('Значення', fontsize=10, fontweight='bold')
    ax.set_title('Реальні дані', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(synth_vals, color=COLOR_ACCENT, linewidth=1, alpha=0.7, label='Синтетичні')
    ax.set_ylabel('Значення', fontsize=10, fontweight='bold')
    ax.set_title('Синтетичні дані', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Гістограми
    ax = axes[1, 0]
    ax.hist(real_vals, bins=50, color=COLOR_PRIMARY, alpha=0.7, 
            edgecolor=COLOR_BLACK, density=True, label='Реальні')
    ax.set_xlabel('Значення', fontsize=10, fontweight='bold')
    ax.set_ylabel('Щільність', fontsize=10, fontweight='bold')
    ax.set_title('Розподіл реальних даних', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.hist(synth_vals, bins=50, color=COLOR_ACCENT, alpha=0.7, 
            edgecolor=COLOR_BLACK, density=True, label='Синтетичні')
    ax.set_xlabel('Значення', fontsize=10, fontweight='bold')
    ax.set_ylabel('Щільність', fontsize=10, fontweight='bold')
    ax.set_title('Розподіл синтетичних даних', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q plots
    from scipy.stats import probplot
    
    ax = axes[2, 0]
    probplot(real_vals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Реальні)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    probplot(synth_vals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Синтетичні)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Порівняння: Реальні vs Синтетичні дані', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()