import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Fashion MNIST
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

X = data.data
y = data.target.astype(int)

# Use smaller dataset for faster execution
X, _, y, _ = train_test_split(
    X, y, train_size=3000, stratify=y, random_state=42
)

# Normalize data
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Different SVM kernels
kernels = ['linear', 'poly', 'rbf']

results = []

for kernel in kernels:

    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    results.append(accuracy)

    print("\nKernel:", kernel)
    print("Accuracy:", accuracy)

# Plot comparison
plt.bar(kernels, results)
plt.xlabel("SVM Kernels")
plt.ylabel("Accuracy")
plt.title("Comparison of SVM Kernels")
plt.show()
