import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
titanic = pd.read_csv("titanic.csv")
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

sns.set_style("whitegrid")

plt.figure(figsize=(7, 5))
sns.histplot(data=titanic, x="Age", bins=20, kde=True)
plt.title("Age Distribution of Titanic Passengers")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=iris, x="species", y="petal_length")
plt.title("Petal Length by Species")
plt.show()

plt.figure(figsize=(7, 5))
sns.violinplot(data=titanic, x="Pclass", y="Age", hue="Sex", split=True)
plt.title("Age by Passenger Class and Sex")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(iris.drop(columns="species").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Iris Features")
plt.show()

sns.pairplot(iris, hue="species")
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(data=titanic, x="Pclass", hue="Survived")
plt.title("Survival Count by Passenger Class")
plt.show()
