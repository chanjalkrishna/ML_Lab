
# Experiment 14 - Activation Functions on MNIST

import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data[:5000]
y = mnist.target[:5000].astype(int)

# Normalize data
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different activation functions
activations = ["logistic", "relu", "tanh"]

for act in activations:

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=act,
        max_iter=20,
        random_state=42
    )

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    y_pred = model.predict(X_test)

    print("\nActivation:", act)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Training Time:", end - start)
