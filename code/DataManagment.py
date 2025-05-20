import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Optional, List, Callable, Tuple
from numpy.typing import NDArray

class DataLoader:
    def __init__(self, data_path='data'):
            self.data_path = data_path
            self.dataset_loaders = {
            'iris': self.load_iris,
            'wine': self.load_wine,
            'seeds': self.load_seeds,
            'banknote': self.load_banknote,
            'heart': self.load_heart,
            'adult': self.load_adult
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
        """Load Wine dataset from local data folder."""
        try:
            path = os.path.join(self.data_path, 'wine.csv')
            print(f"Loading Wine dataset from: {path}")
            df = pd.read_csv(path)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
            target_names = list(np.unique(y))
            return Dataset(
                name='wine',
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                preprocessor=None,  # No preprocessing for this dataset
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
        self.categorical_cols = categorical_cols if categorical_cols else []

    def get_processed_data(self) -> Tuple[NDArray[np.float64], NDArray]:
        if self.preprocessor:
            X_proc = self.preprocessor(self.X)
        else:
            X_proc = self.X
        return X_proc, self.y