import pandas as pd
import matplotlib.pyplot as plt

life = pd.read_csv("life_Expentency.csv", skiprows=3)
life["Year"] = pd.to_numeric(life["Year"], errors="coerce")
life = life.dropna(subset=["Year"])
iris = pd.read_csv("iris.csv")
titanic = pd.read_csv("titanic.csv")
students = pd.read_csv("student_clustering.csv")

plt.figure(figsize=(8, 5))
for country in life["Country"].unique():
    part = life[life["Country"] == country]
    plt.plot(part["Year"], part["Life Expectancy"], marker="o", label=country)
plt.title("Life Expectancy Over Time")
plt.xlabel("Year")
plt.ylabel("Life Expectancy (years)")
plt.legend()
plt.grid(True)
plt.show()

latest = life[life["Year"] == life["Year"].max()]
plt.figure(figsize=(8, 5))
plt.bar(latest["Country"], latest["Life Expectancy"], color=["blue", "orange", "green", "red"])
plt.title("Life Expectancy by Country in " + str(int(life["Year"].max())))
plt.xlabel("Country")
plt.ylabel("Life Expectancy (years)")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(students["cgpa"], students["iq"], color="purple", label="Student")
plt.title("CGPA vs IQ")
plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(iris["sepal_length"], bins=15, color="skyblue", edgecolor="black", label="Sepal Length")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

counts = titanic["Pclass"].value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=["First", "Second", "Third"], autopct="%1.1f%%", startangle=90)
plt.title("Passengers by Class")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for country in life["Country"].unique():
    part = life[life["Country"] == country]
    axes[0, 0].plot(part["Year"], part["Life Expectancy"], label=country)
axes[0, 0].set_title("Life Expectancy")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Years")
axes[0, 0].legend()

axes[0, 1].bar(["Setosa", "Versicolor", "Virginica"],
               iris.groupby("species")["petal_length"].mean(), color="green", label="Mean petal length")
axes[0, 1].set_title("Mean Petal Length")
axes[0, 1].set_ylabel("cm")
axes[0, 1].legend()

axes[1, 0].scatter(iris["petal_length"], iris["petal_width"], color="red", label="Flowers")
axes[1, 0].set_title("Petal Length vs Width")
axes[1, 0].set_xlabel("Petal Length")
axes[1, 0].set_ylabel("Petal Width")
axes[1, 0].legend()

axes[1, 1].hist(students["cgpa"], bins=10, color="orange", edgecolor="black", label="CGPA")
axes[1, 1].set_title("CGPA Distribution")
axes[1, 1].set_xlabel("CGPA")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.tight_layout()
plt.show()
