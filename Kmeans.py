
# Experiment 17 - K-Means on Digits Dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
digits = load_digits()
X = digits.data

# Different number of clusters
k_values = [8, 10, 12]

inertias = []
silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    inertias.append(model.inertia_)

    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    print("\nNumber of Clusters:", k)
    print("Inertia:", model.inertia_)
    print("Silhouette Score:", score)

# Plot Inertia
plt.plot(k_values, inertias, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Plot Silhouette Score
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")
plt.show()
