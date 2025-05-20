import logging
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone

from DataManagment import Dataset
from Classifiers import ClassifierWrapper

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self):
        pass

    def evaluateClassifier(self, dataset: Dataset, classifier: ClassifierWrapper, test_size: float = 0.3, random_state: int = random.randint(0, 9999)) -> float:
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
        logger.info(f"Evaluating classifier on the {dataset.name} dataset with {int(test_size * 100)}% test size...")
        X, y = dataset.get_processed_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        
        logger.info(f"Final test accuracy: {score:.4f}")
        return score
    
    def crossValidateClassifier(self, dataset: Dataset, classifier: ClassifierWrapper, n_splits: int = 5, random_state: int = random.randint(0, 9999)) -> float:
        logger.info(f"Cross-validating classifier on the {dataset.name} dataset with {n_splits} folds...")
        X, y = dataset.get_processed_data()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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

        mean_score = np.mean(scores)
        logger.info(f"Mean cross-validation accuracy: {mean_score:.4f}")
        return mean_score