import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("student_clustering.csv")

score = (df["cgpa"] - df["cgpa"].mean()) / df["cgpa"].std() + (df["iq"] - df["iq"].mean()) / df["iq"].std()
df["placed"] = (score > score.median()).astype(int)
print(df.head())
print(df["placed"].value_counts())

X = df[["cgpa", "iq"]]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

accuracies = []
k_values = range(3, 21, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))
    print("K =", k, "Accuracy =", accuracies[-1])

best_k = k_values[int(np.argmax(accuracies))]
print("Best K:", best_k)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

new_student = scaler.transform(pd.DataFrame([[8.0, 110]], columns=["cgpa", "iq"]))
print("Prediction for new student:", model.predict(new_student)[0])
