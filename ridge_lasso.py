import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
models = {
    "Linear": make_pipeline(StandardScaler(), LinearRegression())
}

# Ridge Regression with CV
ridge = make_pipeline(StandardScaler(), Ridge())
ridge_grid = GridSearchCV(
    ridge, {'ridge__alpha': [0.01, 0.1, 1, 10]},
    cv=5
)
ridge_grid.fit(X_train, y_train)
models["Ridge"] = ridge_grid.best_estimator_

# Lasso Regression with CV
lasso = make_pipeline(StandardScaler(), Lasso(max_iter=10000))
lasso_grid = GridSearchCV(
    lasso, {'lasso__alpha': [0.01, 0.1, 1, 10]},
    cv=5
)
lasso_grid.fit(X_train, y_train)
models["Lasso"] = lasso_grid.best_estimator_

# Train and compare
for name, model in models.items():

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

print("\nBest Ridge Alpha:", ridge_grid.best_params_)
print("Best Lasso Alpha:", lasso_grid.best_params_)
