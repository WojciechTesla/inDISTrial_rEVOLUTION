# Project Plan: Evolving Distance Metrics Using Evolutionary Algorithms for Classification

## ✅ 1. Overview
Design and evolve custom distance metrics using evolutionary algorithms to improve class separation and simplify the hypothesis space for classification models.

---

## ✅ 2. Execution Plan

### ☐ **Phase 1: Background Research & Design**
- ☐ Study metric learning techniques: LMNN, NCA, triplet loss.
- ☐ Review evolutionary algorithms: Genetic Programming, Neuroevolution.
- ☐ Understand cluster analysis metrics: inter/intra-class distance, silhouette score.

### ☐ **Phase 2: Metric Encoding & Objective**
- ☐ Choose encoding strategy:
  - ☐ Mahalanobis matrix (linear, PSD)
  - ☐ Symbolic expression trees (GP)
  - ☐ Neural embeddings (optional) [Maybe just triaining a net a not evolving]
- ☐ Define objective function:
  - ☐ Maximize Fisher criterion: Inter/Intra class ratio
  - ☐ Add complexity regularization (e.g., L2 penalty)

### ☐ **Phase 3: Evolutionary Optimization**
- ☐ Choose framework: DEAP, PyGAD, Nevergrad
- ☐ Implement:
  - ☐ Population initialization
  - ☐ Fitness evaluation on subsampled datasets
  - ☐ Mutation/crossover/selection logic

### ☐ **Phase 4: Evaluation & Baselines**
- ☐ Compare to:
  - ☐ Euclidean, cosine, Mahalanobis (fixed)
  - ☐ LMNN, NCA
  - ☐ Siamese network (optional)
- ☐ Test classifiers:
  - ☐ k-NN
  - ☐ SVM (custom kernel)
  - ☐ Decision tree
- ☐ Evaluate:
  - ☐ Accuracy
  - ☐ Silhouette Score
  - ☐ Model complexity
  - ☐ Generalization gap

### ☐ **Phase 5: Transferability Study**
- ☐ Test metric learned on one dataset in a similar domain on another:
  - ☐ Train on Dataset A, test metric on Dataset B
- ☐ Measure generalization of the metric across datasets

---

## ✅ 3. Dataset Suggestions

### **Numerical Feature Datasets**
- ☐ Iris (UCI)
- ☐ Wine Quality (Red + White variants)
- ☐ [Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

**Same-domain pairs**:
- `Wine Quality - Red` vs `Wine Quality - White`
- `Iris` vs `Seeds Dataset` (UCI) — both are small biological measurements

---

### **Categorical / Mixed Feature Datasets**
- ☐ Adult Census Income
- ☐ Heart Disease (Cleveland)
- ☐ Titanic Survival Dataset

**Same-domain pairs**:
- `Adult Income` vs `KDD Census Income`
- `Heart Disease (Cleveland)` vs `Statlog (Heart)` (UCI)
- `Titanic` vs `Passenger Survival (Sea disasters)` from Kaggle

---

### **Image Feature Datasets (optional)**
- ☐ MNIST (PCA-reduced to 50D)
- ☐ Fashion-MNIST
- ☐ Kuzushiji-MNIST

**Same-domain pairs**:
- `MNIST` vs `Kuzushiji-MNIST` (both handwritten digits)
- `Fashion-MNIST` vs `Clothing1M` (or smaller subsets from DeepFashion)

---

### **Text Feature Datasets**
- ☐ 20 Newsgroups (TF-IDF vectors)
- ☐ SMS Spam Classification
- ☐ AG News Dataset (short texts)

**Same-domain pairs**:
- `20 Newsgroups` vs `Reuters-21578`
- `AG News` vs `Yahoo Answers Topics`

---

## ✅ 4. Tips for Success
- ☐ Start simple (e.g., 2D blobs or Iris) to debug.
- ☐ Visualize transformations (PCA, t-SNE).
- ☐ Compare evolved vs baseline metrics on the same classifier.
- ☐ Use population diversity measures to avoid premature convergence.
- ☐ Log evolution paths, expression trees, and performance progression.
- ☐ Test noise sensitivity and robustness.

---

## ✅ 5. Optional Extensions
- ☐ Incorporate domain-specific priors or constraints (e.g., feature groups).
- ☐ Use learned metrics in downstream pipelines (e.g., clustering or retrieval).
- ☐ Explore zero-shot or few-shot learning using evolved metric spaces.
- ☐ 