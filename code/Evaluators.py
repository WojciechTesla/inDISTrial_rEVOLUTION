import logging
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from typing import Callable, Any
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score

from DataManagment import Dataset
from Classifiers import ClassifierWrapper

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, random_state: int = random.randint(0, 9999)) -> None:
        print(f"Evaluator initialized with random state: {random_state}")
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

        # Summary stats + raw distances
        return {
            "same_class_mean": np.mean(same_class_distances),
            "same_class_std": np.std(same_class_distances),
            "diff_class_mean": np.mean(diff_class_distances),
            "diff_class_std": np.std(diff_class_distances),
            "margin": np.mean(diff_class_distances) - np.mean(same_class_distances),
            "same_class_distances": same_class_distances,
            "diff_class_distances": diff_class_distances
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
        f1s = []
        precisions = []
        recalls = []
        silhouettes = []
        misclassified_per_fold = []
        distance_stats_per_fold = []


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
            y_pred = clf.predict(X_test)
            fold_score = clf.score(X_test, y_test)
            logger.info(f"Fold {fold + 1} accuracy: {fold_score:.4f}")
            scores.append(fold_score)

            f1s.append(f1_score(y_test, y_pred, average='weighted'))
            precisions.append(precision_score(y_test, y_pred, average='weighted'))
            recalls.append(recall_score(y_test, y_pred, average='weighted'))
            try:
                silhouettes.append(silhouette_score(X_test, y_pred))
            except Exception:
                silhouettes.append(float('nan'))

            misclassified = test_idx[y_pred != y_test]
            misclassified_dict = {int(idx): X[idx] for idx in misclassified}
            misclassified_per_fold.append(misclassified_dict)

            if hasattr(clf, 'metric_func'):
                logger.info("Computing distance distribution statistics for metric-based classifier...")
                stats = self.evaluate_distance_distribution(
                    X_test, y_test, clf.metric_func, title=f"Fold {fold+1} Distance Distribution"
                )
                logger.info(f"Fold {fold+1} distance stats: {stats}")
                distance_stats_per_fold.append(stats) 

        mean_score = np.mean(scores)
        mean_f1 = np.nanmean(f1s)
        mean_precision = np.nanmean(precisions)
        mean_recall = np.nanmean(recalls)
        mean_silhouette = np.nanmean(silhouettes)


        logger.warning(f"Mean cross-validation accuracy: {mean_score:.4f}")
        logger.warning(f"Mean F1: {mean_f1:.4f}")
        logger.warning(f"Mean Precision: {mean_precision:.4f}")
        logger.warning(f"Mean Recall: {mean_recall:.4f}")
        logger.warning(f"Mean Silhouette: {mean_silhouette:.4f}")

        return mean_score, misclassified_per_fold, distance_stats_per_fold
    

def show_all_fold_histograms(distance_stats_per_fold):
    n_folds = len(distance_stats_per_fold)
    fig, axes = plt.subplots(1, n_folds, figsize=(6 * n_folds, 5), sharey=True)
    if n_folds == 1:
        axes = [axes]
    for i, stats in enumerate(distance_stats_per_fold):
        ax = axes[i]
        sns.histplot(stats['same_class_distances'], color="blue", label="Same class", stat="density", bins=30, kde=True, ax=ax)
        sns.histplot(stats['diff_class_distances'], color="red", label="Different class", stat="density", bins=30, kde=True, ax=ax)
        ax.set_title(f"Fold {i+1}")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()