from Visualization import *
from DataManagment import DistDataLoader

def testVisualization():
    dataLoader = DistDataLoader("../data")
    datasetsToVisualize = [
        "iris",
        'wine',
        'seeds',
        'banknote',
        'heart',
        # 'adult',
        'synthetic_moons',
    ]

    for datasetName in datasetsToVisualize:
        dataset = dataLoader.load_dataset(datasetName)
        if dataset is None:
            print("Failed to load dataset.")
        else:
            print(f"Loaded dataset: {dataset.name} with {len(dataset.X)} samples.")
            visualize_2d_projection(dataset, title_prefix=f"{datasetName.capitalize()} Dataset", show=True)
            print("Visualization completed.")
