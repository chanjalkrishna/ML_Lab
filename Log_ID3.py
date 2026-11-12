import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Adult dataset
data = fetch_openml('adult', version=2, as_frame=True)

X = data.data
y = data.target

# Split numerical and categorical features
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(exclude='number').columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    ), num_cols),

    ('cat', make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    ), cat_cols)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
lr = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=2000)
)

lr.fit(X_train, y_train)
pred1 = lr.predict(X_test)

# Decision Tree
dt = make_pipeline(
    preprocessor,
    DecisionTreeClassifier(criterion='entropy', max_depth=5)
)

dt.fit(X_train, y_train)
pred2 = dt.predict(X_test)

# Evaluation
for name, pred in [
    ("Logistic Regression", pred1),
    ("Decision Tree", pred2)
]:
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, pos_label='>50K'))
    print("Recall:", recall_score(y_test, pred, pos_label='>50K'))
    print("F1 Score:", f1_score(y_test, pred, pos_label='>50K'))
