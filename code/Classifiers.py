import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from numpy.typing import NDArray
from Metrics import MetricWrapper

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ClassifierWrapper(ABC):
    """
    Abstract base class for classifiers.
    """
    def __init__(self, model: Any) -> None:
        self.model = model

    @abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[Any]) -> None:
        """
        Fit the classifier to the training data.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[Any]:
        """
        Predict the class labels for the input data.
        """
        pass

    @abstractmethod
    def clone(self) -> "ClassifierWrapper":
        """
        Return a fresh copy of the classifier.
        """
        pass

    def score(self, X: NDArray[np.float64], y: NDArray[Any]) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        return accuracy

class KNNWrapper(ClassifierWrapper):
    def __init__(self, metric_func: MetricWrapper, **kwargs: Any) -> None:
        self.metric_func = metric_func
        self.kwargs = kwargs
        super().__init__(KNeighborsClassifier(metric=metric_func, **kwargs))

    def fit(self, X: NDArray[np.float64], y: NDArray[Any]) -> None:
        logger.info("Fitting KNN classifier...")
        self.model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Any]:
        logger.info("Making predictions with KNN classifier...")
        return self.model.predict(X)
    
    def clone(self) -> "KNNWrapper":
        return KNNWrapper(self.metric_func, **self.kwargs)
    
class NearestCentroidWrapper(ClassifierWrapper):
    def __init__(self, metric_func: MetricWrapper, **kwargs: Any) -> None:
        self.metric_func = metric_func
        self.kwargs = kwargs
        super().__init__(NearestCentroid(metric=metric_func, **kwargs))

    def fit(self, X: NDArray[np.float64], y: NDArray[Any]) -> None:
        logger.info("Making predictions with KNN classifier...")
        self.model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Any]:
        logger.info("Making predictions with NearestCentroid classifier...")
        return self.model.predict(X)
    
    def clone(self) -> "NearestCentroidWrapper":
        return NearestCentroidWrapper(self.metric_func, **self.kwargs)
    
class SVMWrapper(ClassifierWrapper):
    def __init__(self, metric_func: MetricWrapper, gamma: float = 1.0, **kwargs: Any) -> None:
        super().__init__(SVC(kernel='precomputed', **kwargs))
        self.metric_func = metric_func
        self.gamma = gamma
        self.X_train_: Optional[NDArray[np.float64]] = None
        self.kwargs = kwargs

    def _compute_kernel(self, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
        logger.debug("Computing kernel matrix...")
        dist_matrix = pairwise_distances(X1, X2, metric=self.metric_func)
        return np.exp(-self.gamma * dist_matrix ** 2)

    def fit(self, X: NDArray[np.float64], y: NDArray[Any]) -> None:
        logger.info("Fitting SVM classifier...")
        self.X_train_ = X
        K_train = self._compute_kernel(X, X)
        self.model.fit(K_train, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Any]:
        if self.X_train_ is None:
            raise ValueError("Model has not been trained yet.")
        logger.info("Making predictions with SVM classifier...")
        K_test = self._compute_kernel(X, self.X_train_)
        return self.model.predict(K_test)
    
    def clone(self) -> "SVMWrapper":
        return SVMWrapper(self.metric_func, gamma=self.gamma, **self.kwargs)
    