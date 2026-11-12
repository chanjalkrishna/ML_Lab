import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Online Retail dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

data = pd.read_excel(url)

# Remove missing customer IDs
data = data.dropna(subset=['CustomerID'])

# Calculate purchase amount
data['Amount'] = data['Quantity'] * data['UnitPrice']

# Remove cancelled transactions and invalid values
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

# Customer-level features
customer = data.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'Amount': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()

customer.columns = ['CustomerID', 'Quantity', 'Amount', 'Purchases']

# Create target labels
customer['Category'] = (
    customer['Amount'] > customer['Amount'].median()
).astype(int)

# Features and target
X = customer[['Quantity', 'Amount', 'Purchases']]
y = customer['Category']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree using entropy
model = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Feature importance
print("Feature Importance:", model.feature_importances_)

# Visualize decision tree
plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Low Value', 'High Value'],
    filled=True
)
plt.show()
