import matplotlib.pyplot as plt

def plot_comprehensive_report(index, y, y_trend, residuals, degree):
    plt.figure(figsize=(10, 5))
    plt.plot(index, y, label='Фактичні дані', marker='.', linestyle='None', alpha=0.6)
    plt.plot(index, y_trend, label=f'Тренд (Ступінь {degree})', color='red', linewidth=2)
    plt.title(f'Апроксимація тренду (Ступінь {degree})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.stem(index, residuals, linefmt='grey', markerfmt='ro', basefmt='k-')
    ax1.set_title('Залишки (часова область)')
    
    ax2.hist(residuals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Розподіл залишків')
    
    plt.tight_layout()
    plt.show()