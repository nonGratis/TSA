
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesClusterer:
    """
    Клас для кластеризації часових рядів.
    Підтримує різні методи та метрики відстаней.
    """
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 3):
        """
        Args:
            method: Метод кластеризації ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Кількість кластерів (для kmeans, hierarchical)
        """
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.model = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.Series, 
                        window_size: int = 24) -> np.ndarray:
        """
        Підготовка ознак для кластеризації через ковзне вікно.
        
        Args:
            data: Часовий ряд
            window_size: Розмір вікна
            
        Returns:
            Матриця ознак (n_windows x window_size)
        """
        clean_data = data.dropna().values
        
        if len(clean_data) < window_size:
            raise ValueError(f"Недостатньо даних. Потрібно мінімум {window_size}")
        
        # Створюємо вікна
        n_windows = len(clean_data) - window_size + 1
        windows = np.array([clean_data[i:i+window_size] 
                           for i in range(n_windows)])
        
        # Нормалізація
        windows_scaled = self.scaler.fit_transform(windows)
        
        return windows_scaled
    
    def prepare_statistical_features(self, data: pd.Series, 
                                     window_size: int = 24) -> np.ndarray:
        """
        Підготовка статистичних ознак для кожного вікна.
        
        Features: mean, std, min, max, median, q25, q75, skew, kurt
        """
        from scipy.stats import skew, kurtosis
        
        clean_data = data.dropna().values
        n_windows = len(clean_data) - window_size + 1
        
        features = []
        for i in range(n_windows):
            window = clean_data[i:i+window_size]
            
            feat = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
                skew(window),
                kurtosis(window)
            ]
            features.append(feat)
        
        features = np.array(features)
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
    
    def cluster_kmeans(self, features: np.ndarray, 
                       n_clusters: Optional[int] = None) -> np.ndarray:
        """K-Means кластеризація."""
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        self.model = KMeans(n_clusters=n_clusters, random_state=42, 
                           n_init=10, max_iter=300)
        self.labels = self.model.fit_predict(features)
        
        return self.labels
    
    def cluster_dbscan(self, features: np.ndarray, 
                       eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """DBSCAN кластеризація (density-based)."""
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        self.labels = self.model.fit_predict(features)
        
        # Кількість знайдених кластерів
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        print(f"  DBSCAN знайшов {n_clusters} кластерів та {n_noise} викидів")
        
        return self.labels
    
    def cluster_hierarchical(self, features: np.ndarray, 
                            n_clusters: Optional[int] = None,
                            linkage_method: str = 'ward') -> np.ndarray:
        """Ієрархічна кластеризація."""
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        self.model = AgglomerativeClustering(n_clusters=n_clusters, 
                                            linkage=linkage_method)
        self.labels = self.model.fit_predict(features)
        
        return self.labels
    
    def cluster(self, data: pd.Series, window_size: int = 24,
                feature_type: str = 'raw') -> Dict:
        """
        Основний метод кластеризації.
        
        Args:
            data: Часовий ряд
            window_size: Розмір вікна
            feature_type: Тип ознак ('raw' або 'statistical')
            
        Returns:
            Словник з результатами кластеризації
        """
        print(f"\n=== КЛАСТЕРИЗАЦІЯ ({self.method.upper()}) ===\n")
        
        # Підготовка ознак
        if feature_type == 'statistical':
            features = self.prepare_statistical_features(data, window_size)
            print(f"  Використано статистичні ознаки (9 ознак)")
        else:
            features = self.prepare_features(data, window_size)
            print(f"  Використано сирі вікна (розмір {window_size})")
        
        print(f"  Кількість вікон: {len(features)}")
        
        # Кластеризація
        if self.method == 'kmeans':
            labels = self.cluster_kmeans(features)
        elif self.method == 'dbscan':
            labels = self.cluster_dbscan(features)
        elif self.method == 'hierarchical':
            labels = self.cluster_hierarchical(features)
        else:
            raise ValueError(f"Невідомий метод: {self.method}")
        
        # Статистика кластерів
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            if label == -1:  # Noise для DBSCAN
                continue
            mask = labels == label
            cluster_stats[int(label)] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(labels) * 100)
            }
        
        print(f"\n  Знайдено кластерів: {len(cluster_stats)}")
        for label, stats in cluster_stats.items():
            print(f"    Кластер {label}: {stats['size']} вікон ({stats['percentage']:.1f}%)")
        
        print("\n" + "="*50 + "\n")
        
        return {
            'labels': labels,
            'features': features,
            'window_size': window_size,
            'n_clusters': len(cluster_stats),
            'cluster_stats': cluster_stats,
            'model': self.model
        }
    
    def calculate_silhouette_score(self, features: np.ndarray, 
                                   labels: np.ndarray) -> float:
        """Обчислення Silhouette Score (якість кластеризації)."""
        from sklearn.metrics import silhouette_score
        
        # Видаляємо noise (-1) для DBSCAN
        mask = labels != -1
        if mask.sum() < 2:
            return -1.0
        
        try:
            score = silhouette_score(features[mask], labels[mask])
            return float(score)
        except:
            return -1.0
    
    def find_optimal_k(self, data: pd.Series, 
                      max_k: int = 10, 
                      window_size: int = 24) -> Dict:
        """
        Пошук оптимальної кількості кластерів через Elbow method.
        
        Returns:
            Словник з inertia та silhouette scores для різних k
        """
        features = self.prepare_features(data, window_size)
        
        inertias = []
        silhouettes = []
        k_range = range(2, max_k + 1)
        
        print(f"\n  Пошук оптимального K (2-{max_k})...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(self.calculate_silhouette_score(features, labels))
        
        # Знаходимо оптимальне k через silhouette
        optimal_k = k_range[np.argmax(silhouettes)]
        
        print(f"  Оптимальне K: {optimal_k} (Silhouette = {max(silhouettes):.3f})")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouettes': silhouettes,
            'optimal_k': optimal_k
        }


def dtw_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Dynamic Time Warping distance між двома часовими рядами.
    """
    distance, _ = fastdtw(ts1, ts2, dist=euclidean)
    return distance


def cluster_with_dtw(data: pd.Series, n_clusters: int = 3, 
                    window_size: int = 24) -> Dict:
    """
    Кластеризація з використанням DTW відстані.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
    
    # Підготовка вікон
    clean_data = data.dropna().values
    n_windows = len(clean_data) - window_size + 1
    windows = np.array([clean_data[i:i+window_size] 
                       for i in range(n_windows)])
    
    # Обчислення DTW матриці відстаней
    print(f"  Обчислення DTW відстаней для {len(windows)} вікон...")
    
    # Для великих датасетів використовуємо вибірку
    if len(windows) > 500:
        step = len(windows) // 500
        windows = windows[::step]
        print(f"  Використано вибірку: {len(windows)} вікон")
    
    distance_matrix = pairwise_distances(windows, metric=dtw_distance, n_jobs=-1)
    
    # Ієрархічна кластеризація
    model = AgglomerativeClustering(n_clusters=n_clusters, 
                                   metric='precomputed',
                                   linkage='average')
    labels = model.fit_predict(distance_matrix)
    
    return {
        'labels': labels,
        'distance_matrix': distance_matrix,
        'windows': windows
    }