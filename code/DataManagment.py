import pandas as pd
import numpy as np
import random
import os
import torch
from collections import defaultdict
from itertools import combinations, product
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from typing import Optional, List, Callable, Tuple
from numpy.typing import NDArray
from torch.utils.data import Dataset as TorchDataset
from sklearn.datasets import make_moons

class DistDataLoader:
    def __init__(self, data_path='data'):
            self.data_path = data_path
            self.dataset_loaders = {
            'iris': self.load_iris,
            'wine': self.load_wine,
            'seeds': self.load_seeds,
            'banknote': self.load_banknote,
            'heart': self.load_heart,
            'adult': self.load_adult,
            'synthetic_moons': self.load_synthetic_moons
        }

    def load_dataset(self, dataset_name):
        """Load and preprocess a specific dataset."""
        if dataset_name in self.dataset_loaders:
            return self.dataset_loaders[dataset_name]()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(self.dataset_loaders.keys())}")
        
    def load_iris(self):
        """Load Iris dataset from local data folder."""
        try:
            path = os.path.join(self.data_path, 'iris.csv')
            print(f"Loading Iris dataset from: {path}")
            df = pd.read_csv(path)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
            target_names = list(np.unique(y))
            return Dataset(
                name='iris',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=None,  # No preprocessing for this dataset
                categorical_cols=None
            )
        except Exception as e:
            print("Iris dataset not found locally. Please download it from UCI repository or run setup.sh script!")
            print("URL: https://archive.ics.uci.edu/ml/datasets/seeds")
            return None
    
    def load_wine(self):
        """Load Wine dataset from local data folder and binarize target into good/bad wine."""
        try:
            path = os.path.join(self.data_path, 'wine.csv')
            print(f"Loading Wine dataset from: {path}")
            df = pd.read_csv(path)

            X = df.iloc[:, :-1].values
            y_raw = df.iloc[:, -1].values

            # Binarize the quality score: Good (>=6) vs Bad (<6)
            y = np.where(y_raw >= 6, 'good wine', 'bad wine')

            feature_names = df.columns[:-1].tolist()
            target_names = ['bad wine', 'good wine']

            return Dataset(
                name='wine',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=None,
                categorical_cols=None
            )
        except Exception as e:
            print("Wine dataset not found locally. Please download it from UCI repository or run setup.sh script!")
            print("URL: https://archive.ics.uci.edu/ml/datasets/wine+quality")
            return None
    
    def load_seeds(self):
        """Load Seeds dataset from local data folder."""
        try:
            path = os.path.join(self.data_path, 'seeds.csv')
            print(f"Loading Seeds dataset from: {path}")
            df = pd.read_csv(path)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values - 1  # Classes are 1-3, convert to 0-2
            feature_names = df.columns[:-1].tolist()
            target_names = list(np.unique(y))
            return Dataset(
                name='seeds',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=None,  # No preprocessing for this dataset
                categorical_cols=None
            )
        except Exception as e:
            print(e)
            print("Seeds dataset not found locally. Please download it from UCI repository or run setup.sh script!")
            print("URL: https://archive.ics.uci.edu/ml/datasets/seeds")
            return None
    
    def load_banknote(self):
        """Load Banknote Authentication dataset from local data folder."""
        try:
            path = os.path.join(self.data_path, 'banknote.csv')
            print(f"Loading Banknote dataset from: {path}")
            df = pd.read_csv(path)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
            target_names = ['Authentic', 'Counterfeit']
            return Dataset(
                name='banknote',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=None,  # No preprocessing for this dataset
                categorical_cols=None
            )
        except Exception as e:
            print("Banknote dataset not found locally. Please download it from UCI repository or run setup.sh script!")
            print("URL: https://archive.ics.uci.edu/ml/datasets/banknote")
            return None
    
    def load_heart(self):
        """Load Heart Disease dataset from local data folder."""
        try:
            path = os.path.join(self.data_path, 'heart.csv')
            print(f"Loading Heart dataset from: {path}")
            df = pd.read_csv(path)

            X = df.drop('target', axis=1).values
            y = df['target'].values
            feature_names = df.drop('target', axis=1).columns.tolist()
            target_names = ['No Disease', 'Disease']
            return Dataset(
                name='heart',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=None,  # No preprocessing for this dataset
                categorical_cols=None
            )
        except Exception as e:
            print("Heart dataset not found locally. Please download it from UCI repository or run setup.sh script!")
            print("URL: https://archive.ics.uci.edu/ml/datasets/heart+Disease")
            return None
    
    def load_adult(self):
        """Load Adult Census Income dataset from local data folder with preprocessing."""
        try:
            # Column names for the Adult dataset
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            
            # Load the data
            path = os.path.join(self.data_path, 'adult.csv')
            print(f"Loading Wine dataset from: {path}")
            df = pd.read_csv(path, header=None, names=column_names, na_values=' ?')
            
            # Identify categorical and numerical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove the target column from feature columns
            if 'income' in categorical_cols:
                categorical_cols.remove('income')
            if 'income' in numerical_cols:
                numerical_cols.remove('income')
            
            # Define preprocessing for numerical and categorical data
            numerical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Bundle preprocessing for numerical and categorical data
            preprocessor = ColumnTransformer(
                transformers=[ 
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            # Extract features and target
            X = df.drop('income', axis=1)
            y = (df['income'] == ' >50K').astype(int)  # 1 if income >50K, 0 otherwise
            
            feature_names = list(X.columns)
            target_names = ['<=50K', '>50K']
            
            return Dataset(
                name='adult',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=preprocessor,
                categorical_cols=categorical_cols
            )
        except Exception as e:
            print("Adult dataset not found locally. Please download it from UCI repository or run setup.sh script!")
            print("URL: https://archive.ics.uci.edu/ml/datasets/adult")
            return None
        
    def load_synthetic_moons(self, noise=0.2, n_samples=200, seed=random.randint(0, 9999)):
        """Generate a non-linearly separable synthetic dataset (moons)."""
        # print(f"Generating synthetic moons dataset with () {n_samples} samples and noise={noise}.")
        print(f"Generating synthetic moons dataset with seed {seed}, {n_samples} samples and noise={noise}.")

        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        feature_names = ['x1', 'x2']
        target_names = ['Class 0', 'Class 1']

        return Dataset(
            name='synthetic_moons',
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            preprocessor=None,
            categorical_cols=None
        )
          
    def load_custom_dataset(self, filename):
        """Load a custom dataset and apply preprocessing."""
        try:
            # Load the dataset
            df = pd.read_csv(filename)

            # Identify categorical and numerical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Check if there's a target column (last column) and exclude it from features
            target_col = df.columns[-1]  # Assume the last column is the target
            feature_names = df.columns[:-1].tolist()
            target_names = [target_col]
            
            # Remove the target column from the feature columns
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            
            # Define preprocessing for numerical and categorical data
            numerical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Bundle preprocessing for numerical and categorical data
            preprocessor = ColumnTransformer(
                transformers=[ 
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            # Extract features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col].values
            
            return Dataset(
                name=os.path.basename(filename),
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=preprocessor,
                categorical_cols=categorical_cols
            )
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            return None
        

class Dataset:
    def __init__(
        self,
        name: str,
        X: NDArray[np.float64],
        y: NDArray,
        feature_names: List[str],
        target_names: List[str],
        preprocessor: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        categorical_cols: Optional[List[int]] = None
    ) -> None:
        self.name = name
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.preprocessor = preprocessor
        self.numOfClasses = len(target_names)
        self.categorical_cols = categorical_cols if categorical_cols else []

        # Label encoding: only if y is not already integer
        if y.dtype.kind in {'U', 'S', 'O'} or not np.issubdtype(y.dtype, np.integer):
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder = None
            self.y = y

    def decode_label(self, y_encoded):
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform([y_encoded])[0]
        return y_encoded

    def get_processed_data(self) -> Tuple[NDArray[np.float64], NDArray]:
        if self.preprocessor:
            X_proc = self.preprocessor(self.X)
        else:
            X_proc = self.X
        return X_proc, self.y
    
class SiameseDataset(TorchDataset):
    def __init__(
        self,
        base_dataset: Dataset,
        num_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        exhaustive: bool = False,
        embeddings: Optional[NDArray[np.float64]] = None,
        hard_negative_threshold: Optional[float] = None
    ):
        self.X, self.y = base_dataset.get_processed_data()
        self.num_pairs = num_pairs or len(self.y)
        self.classes = np.unique(self.y)
        self.class_indices = self._group_by_class()
        self.rng = random.Random(seed)
        self.embeddings = embeddings
        self.hard_negative_threshold = hard_negative_threshold
        self.pairs = self._generate_pairs() if not exhaustive else self.generate_all_pairs(self.X, self.y)

    def _group_by_class(self) -> dict:
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.y):
            class_indices[label].append(idx)
        return class_indices

    def _generate_pairs(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        pairs = []
        attempts = 0
        max_attempts = self.num_pairs * 10  # prevent infinite loop

        while len(pairs) < self.num_pairs and attempts < max_attempts:
            attempts += 1
            if self.rng.random() < 0.5:
                # Positive pair
                label = self.rng.choice(self.classes)
                if len(self.class_indices[label]) < 2:
                    continue
                i1, i2 = self.rng.sample(self.class_indices[label], 2)
                pairs.append((self.X[i1], self.X[i2], 1, self.y[i1], self.y[i2]))
            else:
                # Negative pair
                label1, label2 = self.rng.sample(list(self.classes), 2)
                i1 = self.rng.choice(self.class_indices[label1])    
                i2 = self.rng.choice(self.class_indices[label2])

                # Hard negative filtering
                if self.embeddings is not None and self.hard_negative_threshold is not None:
                    dist = np.linalg.norm(self.embeddings[i1] - self.embeddings[i2])
                    # print(f"Distance between {label1} and {label2}: {dist:.4f}")
                    if dist > self.hard_negative_threshold:
                        continue  # too easy, skip

                pairs.append((self.X[i1], self.X[i2], 0, self.y[i1], self.y[i2]))
        stats = analyze_pair_distances(self.embeddings, pairs)
        print(stats)
        return pairs
    
    def generate_all_pairs(X: NDArray, y: NDArray) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)

        pairs = []

        # Positive pairs
        for indices in class_indices.values():
            for i, j in combinations(indices, 2):
                pairs.append((X[i], X[j], 1, y[i], y[j]))

        # Negative pairs
        labels = list(class_indices.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                for idx1 in class_indices[labels[i]]:
                    for idx2 in class_indices[labels[j]]:
                        pairs.append((X[idx1], X[idx2], 0, y[idx1], y[idx2]))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]

class SiameseTorchDataset(TorchDataset):
    def __init__(self,
                base_dataset: Dataset,
                num_pairs: int = 1000,
                seed: int = 42, 
                embeddings: Optional[np.ndarray] = None,
                hard_negative_threshold: Optional[float] = None):
        self.pairs = SiameseDataset(base_dataset, num_pairs=num_pairs, seed=seed, embeddings=embeddings, hard_negative_threshold=hard_negative_threshold).pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        x1, x2, pair_label, y1, y2 = self.pairs[idx]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(pair_label, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.long),
            torch.tensor(y2, dtype=torch.long)
        )
    

def analyze_pair_distances(embeddings, pairs):
    same_class_distances = []
    diff_class_distances = []

    for x1, x2, pair_label, y1, y2 in pairs:
        # Compute distance between embeddings
        dist = np.linalg.norm(x1 - x2)
        if pair_label == 1:
            same_class_distances.append(dist)
        else:
            diff_class_distances.append(dist)

    # Calculate statistics
    stats = {
        "same_class_mean": np.mean(same_class_distances),
        "same_class_std": np.std(same_class_distances),
        "diff_class_mean": np.mean(diff_class_distances),
        "diff_class_std": np.std(diff_class_distances),
        "same_class_outliers": np.sum(np.abs(same_class_distances - np.mean(same_class_distances)) > 3 * np.std(same_class_distances)),
        "diff_class_outliers": np.sum(np.abs(diff_class_distances - np.mean(diff_class_distances)) > 3 * np.std(diff_class_distances)),
    }

    print("Same class mean:", stats["same_class_mean"])
    print("Same class std:", stats["same_class_std"])
    print("Same class outliers:", stats["same_class_outliers"])
    print("Diff class mean:", stats["diff_class_mean"])
    print("Diff class std:", stats["diff_class_std"])
    print("Diff class outliers:", stats["diff_class_outliers"])

    return stats