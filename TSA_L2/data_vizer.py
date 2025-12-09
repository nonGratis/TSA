from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


COLOR_PRIMARY = '#1323e9'
COLOR_SECONDARY = '#ffaa3a'
COLOR_ACCENT = '#eb5f54'
COLOR_BLACK = '#000000'
COLOR_GRAY = '#cccccc'


class DataVisualizer:

    def __init__(self, output_dir: str = './plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_raw_data(self, t: np.ndarray, y: np.ndarray, imputed: Optional[np.ndarray] = None,
                      title: str = 'Сирі дані часової послідовності') -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all points
        ax.plot(t, y, 'o-', color=COLOR_PRIMARY, linewidth=2, markersize=4,
                label='Спостереження', alpha=0.7)

        # Mark imputed points if provided
        if imputed is not None and imputed.any():
            imputed_idx = np.where(imputed)[0]
            ax.scatter(t[imputed_idx], y[imputed_idx], color=COLOR_ACCENT,
                       s=100, marker='x', linewidths=3, label='Імпутовано', zorder=5)

        ax.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Накопичувальний лічильник (r_id)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / '01_raw_data.svg'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_polynomial_fit(self, t: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                            equation: str, metrics: Dict) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                       gridspec_kw={'height_ratios': [2, 1]})

        # Top: Fit
        ax1.scatter(t, y, color=COLOR_BLACK, s=30, alpha=0.6, label='Спостереження')
        ax1.plot(t, y_pred, color=COLOR_PRIMARY, linewidth=2.5, label='Поліноміальна апроксимація')

        # Add equation and metrics
        textstr = f'{equation}\n\n'
        textstr += f'R² = {metrics["R²"]:.4f}\n'
        textstr += f'RMSE = {metrics["RMSE"]:.2f}\n'
        textstr += f'MAE = {metrics["MAE"]:.2f}'

        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round',
                                                   facecolor='wheat', alpha=0.5))

        ax1.set_ylabel('Накопичувальний лічильник', fontsize=12, fontweight='bold')
        ax1.set_title('Поліноміальна модель (ст. 2)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom: Residuals
        residuals = y - y_pred
        ax2.scatter(t, residuals, color=COLOR_ACCENT, s=30, alpha=0.6)
        ax2.axhline(y=0, color=COLOR_BLACK, linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Залишки', fontsize=12, fontweight='bold')
        ax2.set_title('Аналіз залишків', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / '02_polynomial_fit.svg'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_logistic_fit(self, t: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                         equation: str, metrics: Dict, params: Dict) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # Top: Fit
        ax1.scatter(t, y, color=COLOR_BLACK, s=30, alpha=0.6, label='Спостереження')
        ax1.plot(t, y_pred, color=COLOR_SECONDARY, linewidth=2.5, label='Логістична апроксимація')
        
        # Mark key points
        L = params['L']
        t0_scaled = params['t0']
        
        # Asymptote line
        ax1.axhline(y=L, color=COLOR_GRAY, linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'Асимптота (L={L:.0f})')
        
        # Inflection point (midpoint) as a circle
        half_L = L / 2
        ax1.scatter([t0_scaled], [half_L], color='white', s=150, 
                   marker='o', edgecolors='black', linewidths=2, zorder=10,
                   label=f'Інфлексія (t₀={t0_scaled:.2f}, L/2={half_L:.0f})')
        
        # Add equation and metrics
        textstr = f'{equation}\n\n'
        textstr += f'L (ємність) = {L:.2f}\n'
        textstr += f'k (темп зростання) = {params["k"]:.4f}\n'
        textstr += f't₀ (інфлексія) = {t0_scaled:.2f}\n\n'
        textstr += f'R² = {metrics["R²"]:.4f}\n'
        textstr += f'RMSE = {metrics["RMSE"]:.2f}\n'
        textstr += f'MAE = {metrics["MAE"]:.2f}'
        
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightblue', alpha=0.5))
        
        ax1.set_ylabel('Накопичувальний лічильник', fontsize=12, fontweight='bold')
        ax1.set_title('Логістична модель', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Bottom: Residuals
        residuals = y - y_pred
        ax2.scatter(t, residuals, color=COLOR_ACCENT, s=30, alpha=0.6)
        ax2.axhline(y=0, color=COLOR_BLACK, linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Залишки', fontsize=12, fontweight='bold')
        ax2.set_title('Аналіз залишків', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / '03_logistic_fit.svg'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_extrapolation(self, t_train: np.ndarray, y_train: np.ndarray,
                           t_future: np.ndarray, y_poly: np.ndarray, y_logistic: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(14, 7))

        # Training data
        ax.scatter(t_train, y_train, color=COLOR_BLACK, s=40, alpha=0.6,
                   label='Навчальні дані', zorder=5)

        # Extrapolations
        ax.plot(t_future, y_poly, color=COLOR_PRIMARY, linewidth=2.5,
                linestyle='--', label='Поліном (ст.2)', alpha=0.8)
        ax.plot(t_future, y_logistic, color=COLOR_SECONDARY, linewidth=2.5,
                linestyle='--', label='Логістична', alpha=0.8)

        # Vertical line separating train/test
        train_end = t_train[-1]
        ax.axvline(x=train_end, color=COLOR_ACCENT, linestyle=':', linewidth=2,
                   alpha=0.7, label='Кінець навчання')

        # 95% extrapolation point marker
        extrap_length = len(t_future) - len(t_train)
        t_95 = train_end + int(0.95 * extrap_length)
        ax.axvline(x=t_95, color=COLOR_GRAY, linestyle='-.', linewidth=1.5,
                   alpha=0.5, label='95% екстраполяції')

        # Final values info box
        final_poly = y_poly[-1]
        final_log = y_logistic[-1]

        infostr = f'Кінцеві значення (t={t_future[-1]:.0f}):\n'
        infostr += f'Поліном: {final_poly:.0f} (зростає)\n'
        infostr += f'Логістична: {final_log:.0f}\n'
        infostr += f'Δ = {abs(final_poly - final_log):.0f}'

        ax.text(0.02, 0.02, infostr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        ax.set_xlabel('Індекс часу (години)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Накопичувальний лічильник', fontsize=12, fontweight='bold')
        ax.set_title('Екстраполяційне порівняння: дивергенція полінома vs асимптота логістики',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add annotation
        textstr = 'Поліноміальні апроксимації демонструють дивергенцію; логістична модель має асимптоту L.'
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        filepath = self.output_dir / '04_extrapolation_comparison.svg'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_residuals_histogram(self, residuals_poly: np.ndarray, residuals_logistic: np.ndarray) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Polynomial residuals
        ax1.hist(residuals_poly, bins=30, color=COLOR_PRIMARY, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=2)
        ax1.set_xlabel('Значення залишку', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Частота', fontsize=11, fontweight='bold')
        ax1.set_title('Розподіл залишків (поліном)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        mean_poly = np.mean(residuals_poly)
        std_poly = np.std(residuals_poly)
        ax1.text(0.02, 0.98, f'μ = {mean_poly:.2f}\nσ = {std_poly:.2f}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Logistic residuals
        ax2.hist(residuals_logistic, bins=30, color=COLOR_SECONDARY, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color=COLOR_ACCENT, linestyle='--', linewidth=2)
        ax2.set_xlabel('Значення залишку', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Частота', fontsize=11, fontweight='bold')
        ax2.set_title('Розподіл залишків (логістична)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        mean_log = np.mean(residuals_logistic)
        std_log = np.std(residuals_logistic)
        ax2.text(0.02, 0.98, f'μ = {mean_log:.2f}\nσ = {std_log:.2f}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        filepath = self.output_dir / '05_residuals_histogram.svg'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    
