import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
data = fetch_20newsgroups(
    subset='train',
    categories=['sci.space', 'rec.sport.baseball'],
    remove=('headers', 'footers', 'quotes')
)

# Convert text into word counts
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data.data)

# Total word counts
counts = np.asarray(X.sum(axis=0)).ravel()

# MLE estimation
mle = counts / counts.sum()

print("MLE Probabilities:")
print(mle[:10])

# MAP estimation using Dirichlet prior
priors = [0.1, 1, 10]

for alpha in priors:

    map_estimate = (counts + alpha - 1)
    map_estimate = np.maximum(map_estimate, 0)
    map_estimate = map_estimate / map_estimate.sum()

    print("\nDirichlet Prior:", alpha)
    print("MAP Probabilities:")
    print(map_estimate[:10])
