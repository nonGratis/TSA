import numpy as np
from collections import deque
from typing import Optional


class AdaptiveQ:
    """
    Адаптивна оцінка процесного шуму Q на основі нормалізованих інновацій
    
    Стратегія:
    - Зберігає скользяче вікно нормалізованих залишків (residual / sqrt(S))
    - При зростанні дисперсії → збільшує Q (фільтр стає менш жорстким)
    - При стабільній дисперсії → поступово зменшує Q (фільтр стає жорсткішим)
    - Використовує інноваційну коваріацію S для правильної нормалізації
    
    Параметри:
    - window: Розмір скользячого вікна
    - q_min, q_max: Межі для Q
    - adapt_rate: Коефіцієнт адаптації (множник)
    - alpha: Поріг для визначення "різкого зростання" варіації
    """
    
    def __init__(
        self,
        window: int = 24,
        q_min: float = 1e-6,
        q_max: float = 1e2,
        adapt_rate: float = 1.2,
        alpha: float = 3.0,
        init_q: float = 1.0
    ):
        """
        Args:
            window: Розмір скользячого вікна залишків (год)
            q_min: Мінімальне значення Q
            q_max: Максимальне значення Q
            adapt_rate: Швидкість адаптації (множник)
            alpha: Поріг для визначення аномальної варіації
            init_q: Початкове значення Q
        """
        self.window = window
        self.q_min = q_min
        self.q_max = q_max
        self.adapt_rate = adapt_rate
        self.alpha = alpha
        
        # Поточне значення Q
        self.q_current = max(min(init_q, q_max), q_min)
        
        # Скользяче вікно нормалізованих інновацій
        self.normalized_innovations = deque(maxlen=window)
        
        # Статистика для адаптації
        self.variance_history = deque(maxlen=10)  # Історія дисперсій
        self.median_variance: Optional[float] = None
        
        # Лічильники для логування
        self.increase_count = 0
        self.decrease_count = 0
        self.max_reached_count = 0
        self.min_reached_count = 0
    
    def update(self, residual: float, innovation_covariance: float = 1.0) -> float:
        """
        Оновлення Q на основі нового залишку та інноваційної коваріації
        
        Args:
            residual: Новий залишок (measurement - prediction)
            innovation_covariance: S = H P H^T + R (коваріація інновації)
            
        Returns:
            Оновлене значення Q
        """
        # Нормалізуємо залишок по sqrt(S) для правильної статистики
        if innovation_covariance > 1e-10:
            normalized_residual = residual / np.sqrt(innovation_covariance)
        else:
            normalized_residual = residual
        
        # Додаємо нормалізований залишок
        self.normalized_innovations.append(normalized_residual)
        
        # Потрібно мінімум 2 точки для обчислення дисперсії
        if len(self.normalized_innovations) < 2:
            return self.q_current
        
        # Обчислюємо поточну дисперсію вікна (має бути ~1 для правильно налаштованого фільтра)
        current_variance = float(np.var(self.normalized_innovations, ddof=1))
        self.variance_history.append(current_variance)
        
        # Потрібно мінімум кілька вимірів для обчислення медіани
        if len(self.variance_history) < 3:
            return self.q_current
        
        # Обчислюємо медіану дисперсії
        self.median_variance = float(np.median(self.variance_history))
        
        # Перевірка на різке зростання дисперсії (фільтр не встигає)
        if current_variance > self.alpha * max(self.median_variance, 1.0):
            # Збільшуємо Q (послаблюємо фільтр)
            new_q = self.q_current * self.adapt_rate
            
            if new_q >= self.q_max:
                new_q = self.q_max
                self.max_reached_count += 1
                
                if self.max_reached_count % 10 == 1:  # Логуємо кожні 10 разів
                    print(f"WARN: Адаптивний Q досягнув максимуму q_max={self.q_max:.2e}")
            else:
                self.increase_count += 1
            
            self.q_current = new_q
        
        # Перевірка на стабільність (дисперсія близька до очікуваної ~1)
        elif current_variance < 0.7 * max(self.median_variance, 0.8) and self.q_current > self.q_min:
            # Зменшуємо Q (посилюємо фільтр)
            new_q = self.q_current / self.adapt_rate
            
            if new_q <= self.q_min:
                new_q = self.q_min
                self.min_reached_count += 1
            else:
                self.decrease_count += 1
            
            self.q_current = new_q
        
        return self.q_current
    
    def get_q(self) -> float:
        """Повернути поточне значення Q"""
        return self.q_current
    
    def get_statistics(self) -> dict:
        """
        Повернути статистику адаптації
        
        Returns:
            Словник зі статистикою
        """
        return {
            'q_current': self.q_current,
            'q_min': self.q_min,
            'q_max': self.q_max,
            'median_variance': self.median_variance,
            'window_size': len(self.normalized_innovations),
            'increase_count': self.increase_count,
            'decrease_count': self.decrease_count,
            'max_reached_count': self.max_reached_count,
            'min_reached_count': self.min_reached_count
        }
    
    def reset(self, init_q: Optional[float] = None) -> None:
        """
        Скидання адаптера до початкового стану
        
        Args:
            init_q: Нове початкове значення Q (якщо None - використовує поточне)
        """
        if init_q is not None:
            self.q_current = max(min(init_q, self.q_max), self.q_min)
        
        self.normalized_innovations.clear()
        self.variance_history.clear()
        self.median_variance = None
        
        self.increase_count = 0
        self.decrease_count = 0
        self.max_reached_count = 0
        self.min_reached_count = 0
