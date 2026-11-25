import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("student_clustering.csv")
print(df.head())

X = StandardScaler().fit_transform(df[["cgpa", "iq"]])

inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.show()

K = 3
model = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X)

plt.figure(figsize=(7, 5))
plt.scatter(df["cgpa"], df["iq"], c=df["cluster"], cmap="viridis")
centers_original = StandardScaler().fit(df[["cgpa", "iq"]]).inverse_transform(model.cluster_centers_)
plt.scatter(centers_original[:, 0], centers_original[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.title("Student Clusters")
plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.legend()
plt.show()

summary = df.groupby("cluster").agg(
    students=("cgpa", "count"),
    avg_cgpa=("cgpa", "mean"),
    avg_iq=("iq", "mean"),
    min_cgpa=("cgpa", "min"),
    max_cgpa=("cgpa", "max"),
    min_iq=("iq", "min"),
    max_iq=("iq", "max"))
print(summary.round(2))

for c in summary.index:
    row = summary.loc[c]
    cg = "high" if row["avg_cgpa"] >= 7.5 else ("medium" if row["avg_cgpa"] >= 6 else "low")
    iq = "high" if row["avg_iq"] >= 108 else ("medium" if row["avg_iq"] >= 95 else "low")
    print("Cluster", c, ":", int(row["students"]), "students with", cg, "CGPA and", iq, "IQ")
