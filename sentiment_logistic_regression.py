import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("sentiment_analysis_sample_100000.csv")
df = df.drop_duplicates(subset="sentence")
print(df.shape)
print(df["sentiment"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["sentence"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

reviews = ["This product is excellent and very useful", "This was a terrible and awful experience"]
print(model.predict(vectorizer.transform(reviews)))
