import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = load_iris()

# Use first two features
X = data.data[:, :2]

# Binary classification: Setosa vs Non-Setosa
y = (data.target == 0).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear SVM
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    facecolors='none',
    edgecolors='red',
    s=100,
    label='Support Vectors'
)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Linear SVM Decision Boundary')
plt.legend()
plt.show()
