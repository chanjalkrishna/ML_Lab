from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
data = fetch_20newsgroups(
    subset='all',
    categories=['sci.space', 'rec.sport.baseball',
                'comp.graphics', 'talk.politics.misc']
)

# Convert text into numerical data
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data.data)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

pred1 = mnb.predict(X_test)

print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1, average='weighted'))

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

pred2 = bnb.predict(X_test)

print("\nBernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2, average='weighted'))
