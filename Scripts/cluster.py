
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class ClusterMetrics:

    inertia: float
    silhouette: Optional[float]
    accuracy: Optional[float] = None

@dataclass
class PokerHandClusterer:

    n_clusters: Optional[int] = None
    random_state: int = 42
    init: str = "k-means++"
    max_iter: int = 300
    model: KMeans = field(init=False)
    _cluster_label_map: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.model = KMeans(
            n_clusters=self.n_clusters if self.n_clusters is not None else 10,
            init=self.init,
            n_init="auto",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    @property
    def cluster_label_map(self) -> Dict[int, int]:

        return dict(self._cluster_label_map)

    def fit(self, features: np.ndarray) -> None:

        if features.size == 0:
            raise ValueError("Feature matrix is empty; provide at least one sample.")
        self.model.fit(features)

    def predict(self, features: np.ndarray) -> np.ndarray:

        if not hasattr(self.model, "cluster_centers_"):
            raise RuntimeError("Model must be fitted before predicting clusters.")
        return self.model.predict(features)

    def build_label_map(self, features: np.ndarray, labels: np.ndarray) -> None:

        predictions = self.predict(features)
        mapping: Dict[int, int] = {}
        for cluster_id in np.unique(predictions):
            cluster_labels = labels[predictions == cluster_id]
            if cluster_labels.size == 0:
                continue
            most_common = Counter(cluster_labels).most_common(1)[0][0]
            mapping[int(cluster_id)] = int(most_common)
        self._cluster_label_map = mapping

    def predict_labels(self, features: np.ndarray) -> np.ndarray:

        if not self._cluster_label_map:
            raise RuntimeError(
                "Cluster-to-label mapping has not been computed. Call 'build_label_map'"
                " with labelled data first."
            )
        predictions = self.predict(features)
        return np.vectorize(self._cluster_label_map.get)(predictions)

    def evaluate(
        self,
        features: np.ndarray,
        *,
        labels: Optional[np.ndarray] = None,
    ) -> ClusterMetrics:

        inertia = float(self.model.inertia_)
        silhouette = None
        if len(features) > 1 and len(set(self.model.labels_)) > 1:
            silhouette = float(
                silhouette_score(features, self.model.labels_, metric="euclidean")
            )

        accuracy = None
        if labels is not None and self._cluster_label_map:
            predicted = self.predict_labels(features)
            accuracy = float(np.mean(predicted == labels))

        return ClusterMetrics(inertia=inertia, silhouette=silhouette, accuracy=accuracy)

    def save_model(self, filepath: str | Path) -> None:
        Salva o modelo treinado em um arquivo pickle.
        
        Parameters
        ----------
        filepath : str | Path
            Caminho onde o modelo será salvo.
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'cluster_label_map': self._cluster_label_map,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'init': self.init,
            'max_iter': self.max_iter
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str | Path) -> PokerHandClusterer:
        Carrega um modelo previamente treinado de um arquivo pickle.
        
        Parameters
        ----------
        filepath : str | Path
            Caminho do arquivo do modelo salvo.
            
        Returns
        -------
        PokerHandClusterer
            Instância do modelo carregado pronta para fazer predições.
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            n_clusters=model_data['n_clusters'],
            random_state=model_data['random_state'],
            init=model_data['init'],
            max_iter=model_data['max_iter']
        )
        
        instance.model = model_data['model']
        instance._cluster_label_map = model_data['cluster_label_map']
        
        return instance