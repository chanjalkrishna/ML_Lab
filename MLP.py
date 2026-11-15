
# Experiment 13 - MLP on Wine Quality Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("winequality-red.csv", sep=";")

# Separate input and output
X = data.drop("quality", axis=1)
y = data["quality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Different MLP architectures
models = [
    (10,),
    (50,),
    (50, 30)
]

for layers in models:

    model = MLPRegressor(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nHidden Layers:", layers)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
