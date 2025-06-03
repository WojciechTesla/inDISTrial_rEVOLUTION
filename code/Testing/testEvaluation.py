from DataManagment import DistDataLoader
from Evaluators import Evaluator
from Classifiers import KNNWrapper, SVMWrapper, NearestCentroidWrapper
from Metrics import EuclideanMetric, CosineDistanceMetric, SiameseNetworkMetric
from Models import SiameseNetwork
from Training import train_siamese
import pandas as pd
import torch
import os


def get_siamese_model_for_dataset(dataset, input_dim, embedding_dim=128, model_path=None):
    if model_path is None:
        model_path = f"../models/siamese_model_{dataset.name}.pth"
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    if os.path.exists(model_path):
        print(f"🔍 Loading SiameseNetwork from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print(f"⚠️ {model_path} not found. Training new SiameseNetwork for {dataset.name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_siamese(model, dataset, optimizer)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model

def get_metric(metric_name, dataset):
    if metric_name == "SiameseNetwork":
        input_dim = dataset.X.shape[1] if hasattr(dataset.X, "shape") else len(dataset.X[0])
        model_path = f"../models/siamese_model_{dataset.name}.pth"
        siamese_model = get_siamese_model_for_dataset(dataset, input_dim, model_path=model_path)
        return SiameseNetworkMetric(model=siamese_model)
    elif metric_name == "Euclidean":
        return EuclideanMetric()
    elif metric_name == "Cosine":
        return CosineDistanceMetric()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

# Define available models and metrics
metric_funcs = {
    "Euclidean": EuclideanMetric(),
    "Cosine": CosineDistanceMetric(),
    "SiameseNetwork": None,
}

classifier_classes = {
    "KNN": lambda metric: KNNWrapper(metric_func=metric, n_neighbors=10),
    "SVM": lambda metric: SVMWrapper(metric_func=metric),
    "NearestCentroid": lambda metric: NearestCentroidWrapper(metric_func=metric),
}

dataset_names = [
    "iris",
    "wine",
    "seeds",
    "banknote",
    "heart",
    # "adult",
    "synthetic_moons",
]


def evaluate_model_metric_all_datasets(model_name: str, metric_name: str):
    evaluator = Evaluator()
    loader = DistDataLoader("../data")
    results = []

    model_func = classifier_classes[model_name]

    for name in dataset_names:
        dataset = loader.load_dataset(name)
        if dataset is None:
            print(f"❌ Failed to load {name}")
            continue

        metric_func = get_metric(metric_name, dataset)
        model = model_func(metric_func)
        score, misclassified_per_fold, distDistribution  = evaluator.crossValidateClassifier(dataset, model)
        results.append({"Dataset": name, "Model": model_name, "Metric": metric_name, "Accuracy": score})

    return pd.DataFrame(results)


def evaluate_all_metrics_on_dataset(model_name: str, dataset_name: str):
    evaluator = Evaluator()
    loader = DistDataLoader("../data")
    dataset = loader.load_dataset(dataset_name)

    if dataset is None:
        print(f"❌ Failed to load {dataset_name}")
        return pd.DataFrame()

    results = []
    for metric_name in metric_funcs.keys():
        metric_func = get_metric(metric_name, dataset)
        model = classifier_classes[model_name](metric_func)
        score, misclassified_per_fold, distDistribution  = evaluator.crossValidateClassifier(dataset, model)
        results.append({"Dataset": dataset_name, "Model": model_name, "Metric": metric_name, "Accuracy": score})

    return pd.DataFrame(results)

def evaluate_all_models_on_dataset(metric_name: str, dataset_name: str):
    evaluator = Evaluator()
    loader = DistDataLoader("../data")
    dataset = loader.load_dataset(dataset_name)

    if dataset is None:
        print(f"❌ Failed to load {dataset_name}")
        return pd.DataFrame()

    metric_func = get_metric(metric_name, dataset)
    results = []

    for model_name, model_func in classifier_classes.items():
        model = model_func(metric_func)
        score, misclassified_per_fold, distDistribution  = evaluator.crossValidateClassifier(dataset, model)
        results.append({"Dataset": dataset_name, "Model": model_name, "Metric": metric_name, "Accuracy": score})

    return pd.DataFrame(results)


def evaluate_all_combinations():
    evaluator = Evaluator()
    loader = DistDataLoader("../data")
    results = []

    for dataset_name in dataset_names:
        dataset = loader.load_dataset(dataset_name)
        if dataset is None:
            print(f"❌ Failed to load {dataset_name}")
            continue

        for model_name, model_func in classifier_classes.items():
            for metric_name in metric_funcs.keys():
                metric_func = get_metric(metric_name, dataset)
                model = model_func(metric_func)
                score, misclassified_per_fold, distDistribution  = evaluator.crossValidateClassifier(dataset, model)
                results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Metric": metric_name,
                    "Accuracy": score
                })

    df = pd.DataFrame(results)
    return df

def evaluate_combination(model_name: str, dataset_name: str, metric_name: str, n_splits: int = 5):
    """
    Evaluates a given classifier on a dataset with a specific metric.
    Returns mean accuracy and misclassified indices per fold.
    """
    evaluator = Evaluator()
    loader = DistDataLoader("../data")
    dataset = loader.load_dataset(dataset_name)
    if dataset is None:
        print(f"❌ Failed to load {dataset_name}")
        return None, None

    metric_func = get_metric(metric_name, dataset)
    model = classifier_classes[model_name](metric_func)
    score, misclassified_per_fold, distDistribution = evaluator.crossValidateClassifier(dataset, model, n_splits=n_splits)
    # print(f"Mean accuracy: {score:.4f}")
    # print(f"Misclassified indices per fold: {misclassified_per_fold}")
    return score, misclassified_per_fold, distDistribution


def summarize_results(df: pd.DataFrame):
    print("\n📊 Accuracy Pivot Table:")
    print(df.pivot_table(index=["Dataset"], columns=["Model", "Metric"], values="Accuracy"))

def testEvaluation():
    # Example usage:

    # 1. Single model+metric across all datasets
    # df1 = evaluate_model_metric_all_datasets("KNN", "Euclidean")

    # 2. All metrics for a model on one dataset
    df2 = evaluate_all_metrics_on_dataset("KNN", "iris")

    # 3. All models with one metric on one dataset
    # df3 = evaluate_all_models_on_dataset("Manhattan", "wine")

    # 4. Full cross-product
    # df_all = evaluate_all_combinations()

    # Summary printout
    summarize_results(df2)

    # Optional: Save all results
    # df_all.to_csv("evaluation_results.csv", index=False)
    print("Evaluation tests completed successfully.")

if __name__ == "__main__":
    testEvaluation()
