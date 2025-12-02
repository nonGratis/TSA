import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'e-Ukraine'

COLOR_PRIMARY = '#1323e9'
COLOR_SECONDARY = '#ffaa3a'
COLOR_ACCENT = '#eb5f54'
COLOR_BLACK = '#000000'

def plot_report(index, y, y_trend, residuals, y_synthetic, residuals_synthetic, model_type, coeffs):
    if model_type == 'poly':
        degree = len(coeffs) - 1
        terms = []
        for i, coef in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f'{coef:.2f}')
            elif power == 1:
                terms.append(f'{coef:.2f}t')
            else:
                terms.append(f'{coef:.2f}t^{power}')
        equation = 'y = ' + ' + '.join(terms).replace('+ -', '- ')
    else:
        equation = f'y = {coeffs[0]:.2f}·ln(t) + {coeffs[1]:.2f}'
    
    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(10, 12))
    
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.plot(index, y, label='Фактичні дані', marker='.', linestyle='None', color=COLOR_PRIMARY, alpha=0.6, zorder=2)
    ax1.plot(index, y_trend, label=f'Модель тренду', color=COLOR_SECONDARY, linewidth=2, zorder=3)
    ax1.set_title(f'Апроксимація процесу\n{equation}', fontsize=10)
    ax1.set_xlabel('Датачас')
    ax1.set_ylabel('Кіл-сть відповідей (кумулятивно)')
    ax1.legend()
    
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.stem(index, residuals, linefmt='grey', markerfmt='o', basefmt='k-')
    ax2.get_children()[0].set_color(COLOR_SECONDARY)
    ax2.set_title('Залишки теоретичної моделі до даних')
    ax2.set_xlabel('Датачас')
    ax2.set_ylabel('Абсалютне відхилення')
    
    ax3.grid(True, alpha=0.3, axis='y', zorder=0)
    ax3.hist(residuals, bins='auto', color=COLOR_SECONDARY, edgecolor=COLOR_BLACK, alpha=0.7, zorder=2)
    ax3.set_title('Гістограма розподілу залишків теоретичної моделі')
    ax3.set_xlabel('Величина похибки')
    ax3.set_ylabel('Частота')
    
    ax4.grid(True, alpha=0.3, zorder=0)
    ax4.plot(index, y, color=COLOR_PRIMARY, linestyle='None', marker='o', markersize=4, alpha=0.8, label='Фактичні дані')
    ax4.plot(index, y_synthetic, color=COLOR_ACCENT, linewidth=2, alpha=0.7, label='Синтезовані дані')
    ax4.set_title('Порівняння фактичних та синтезованих даних')
    ax4.set_xlabel('Датачас')
    ax4.set_ylabel('Кіл-сть відповідей (кумулятивно)')
    ax4.legend()
    
    ax5.grid(True, alpha=0.3, zorder=0)
    ax5.stem(index, residuals_synthetic, linefmt='grey', markerfmt='o', basefmt='k-')
    ax5.get_children()[0].set_color(COLOR_ACCENT)
    ax5.set_title('Залишки синтезованої моделі до тренду')
    ax5.set_xlabel('Датачас')
    ax5.set_ylabel('Абсолютне відхилення')
    
    ax6.grid(True, alpha=0.3, axis='y', zorder=0)
    ax6.hist(residuals_synthetic, bins='auto', color=COLOR_ACCENT, edgecolor=COLOR_BLACK, alpha=0.7, zorder=2)
    ax6.set_title('Гістограма розподілу залишків синтезованих даних')
    ax6.set_xlabel('Величина похибки')
    ax6.set_ylabel('Частота')
    
    plt.tight_layout()
    plt.show()