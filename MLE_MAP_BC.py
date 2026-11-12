from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLE
mle = LogisticRegression(penalty=None, max_iter=5000)
mle.fit(X_train, y_train)

# MAP - L1
map_l1 = LogisticRegression(
    penalty='l1', solver='liblinear', C=1, max_iter=5000
)
map_l1.fit(X_train, y_train)

# MAP - L2
map_l2 = LogisticRegression(
    penalty='l2', C=1, max_iter=5000
)
map_l2.fit(X_train, y_train)

# Evaluation
for name, model in [
    ("MLE", mle),
    ("MAP L1", map_l1),
    ("MAP L2", map_l2)
]:
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Coefficients:", model.coef_[0][:5])
