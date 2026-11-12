import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Fashion MNIST
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

X = data.data
y = data.target.astype(int)

# Use a smaller sample for faster execution
X, _, y, _ = train_test_split(
    X, y, train_size=10000, stratify=y, random_state=42
)

# Normalize pixel values
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different K values
for k in [3, 5, 7]:

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nK =", k)
    print("Accuracy:", accuracy_score(y_test, pred))

# Display sample image
plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.title("Fashion MNIST Image")
plt.show()
