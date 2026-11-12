import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X = data.data[:, [5]]  # Average number of rooms
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias column
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# 1. Gradient Descent
theta = np.zeros(2)
lr = 0.01
iterations = 1000
m = len(y_train)

for i in range(iterations):
    predictions = X_train_b @ theta
    gradients = (2/m) * X_train_b.T @ (predictions - y_train)
    theta = theta - lr * gradients

y_pred_gd = X_test_b @ theta

# 2. Normal Equation
theta_normal = np.linalg.pinv(X_train_b) @ y_train
y_pred_normal = X_test_b @ theta_normal

# Evaluation
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2:", r2_score(y_test, y_pred_gd))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, y_pred_normal))
print("R2:", r2_score(y_test, y_pred_normal))

# Visualization
plt.scatter(X_test, y_test)
order = np.argsort(X_test[:, 0])
plt.plot(X_test[order], y_pred_gd[order], color='red')
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("Linear Regression")
plt.show()
