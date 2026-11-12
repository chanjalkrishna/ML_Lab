import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

# Remove missing values
data = data.dropna()

X = data[['displacement']]
y = data['mpg']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Polynomial Regression
for degree in [2, 3, 4]:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    pred = model.predict(X_test_poly)

    print("\nPolynomial Degree:", degree)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

    # Plot
    x_plot = np.linspace(X.min().iloc[0],
                         X.max().iloc[0], 100).reshape(-1, 1)

    y_plot = model.predict(poly.transform(x_plot))

    plt.scatter(X, y, alpha=0.3)
    plt.plot(x_plot, y_plot)
    plt.xlabel("Displacement")
    plt.ylabel("MPG")
    plt.title(f"Polynomial Regression Degree {degree}")
    plt.show()
