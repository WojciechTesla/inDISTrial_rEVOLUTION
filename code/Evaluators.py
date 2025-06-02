import logging
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from typing import Callable, Any

from DataManagment import Dataset
from Classifiers import ClassifierWrapper

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, random_state: int = random.randint(0, 9999)) -> None:
        self.random_state = random_state
        pass

    def evaluate_distance_distribution(
        self, X: np.ndarray, y: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float], title: str = ""
    ) -> dict:
        same_class_distances = []
        diff_class_distances = []

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dist = metric(X[i], X[j])
                if y[i] == y[j]:
                    same_class_distances.append(dist)
                else:
                    diff_class_distances.append(dist)

        # Visualization (optional)
        sns.histplot(same_class_distances, color="blue", label="Same class", stat="density", bins=30, kde=True)
        sns.histplot(diff_class_distances, color="red", label="Different class", stat="density", bins=30, kde=True)
        plt.title(title or "Distance Distribution")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Summary stats
        return {
            "same_class_mean": np.mean(same_class_distances),
            "same_class_std": np.std(same_class_distances),
            "diff_class_mean": np.mean(diff_class_distances),
            "diff_class_std": np.std(diff_class_distances),
            "margin": np.mean(diff_class_distances) - np.mean(same_class_distances)
        }

    def evaluateClassifier(self, dataset: Dataset, classifier: ClassifierWrapper, test_size: float = 0.3) -> float:
        """
        Evaluate the classifier on the given dataset using a train/test split.
        
        Parameters:
            dataset (Dataset): The dataset object containing data and labels.
            classifier (ClassifierWrapper): The classifier to evaluate.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.
        
        Returns:
            float: Accuracy score on the test set.
        """
        logger.info(f"Evaluating classifier (seed={self.random_state}) on the {dataset.name} dataset with {int(test_size * 100)}% test size...")
        X, y = dataset.get_processed_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, shuffle=True)

        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        
        logger.info(f"Final test accuracy: {score:.4f}")
        return score
    
    def crossValidateClassifier(self, dataset: Dataset, classifier: ClassifierWrapper, n_splits: int = 5) -> float:
        logger.info(f"Cross-validating classifier (seed={self.random_state}) on the {dataset.name} dataset with {n_splits} folds...")
        X, y = dataset.get_processed_data()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold + 1}/{n_splits}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if dataset.preprocessor is not None:
                logger.info("Applying preprocessing for this fold...")
                preprocessor = clone(dataset.preprocessor)
                preprocessor.fit(X_train)
                X_train = preprocessor.transform(X_train)
                X_test = preprocessor.transform(X_test)

            clf = classifier.clone()
            clf.fit(X_train, y_train)
            fold_score = clf.score(X_test, y_test)
            logger.info(f"Fold {fold + 1} accuracy: {fold_score:.4f}")
            scores.append(fold_score)

            if hasattr(clf, 'metric_func'):
                logger.info("Computing distance distribution statistics for metric-based classifier...")
                stats = self.evaluate_distance_distribution(
                    X_test, y_test, clf.metric_func, title=f"Fold {fold+1} Distance Distribution"
                )
                logger.info(f"Fold {fold+1} distance stats: {stats}")

        mean_score = np.mean(scores)
        logger.info(f"Mean cross-validation accuracy: {mean_score:.4f}")
        return mean_score