import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

print("Missing values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
print("Missing values after cleaning:")
print(df.isnull().sum())

print("Female passengers:")
print(df[df["Sex"] == "female"].head())
print("Passengers older than 60:")
print(df[df["Age"] > 60][["Name", "Age", "Survived"]].head())
print("First class males who survived:")
print(df[(df["Pclass"] == 1) & (df["Sex"] == "male") & (df["Survived"] == 1)].head())

print("Survival rate by sex:")
print(df.groupby("Sex")["Survived"].mean())
print("Average fare by class:")
print(df.groupby("Pclass")["Fare"].mean())
print("Survival rate by class and sex:")
print(df.groupby(["Pclass", "Sex"])["Survived"].mean())
print(df.groupby("Embarked").agg({"Fare": ["mean", "max"], "Age": "mean"}))

class_info = pd.DataFrame({
    "Pclass": [1, 2, 3],
    "ClassName": ["First", "Second", "Third"]
})
merged = pd.merge(df, class_info, on="Pclass", how="left")
print(merged[["Name", "Pclass", "ClassName"]].head())

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 60, 100],
                        labels=["Child", "Teen", "Adult", "Senior"])
print(df[["Age", "AgeGroup", "FamilySize"]].head())
print(df.sort_values("Fare", ascending=False).head())

summary = df.groupby("AgeGroup", observed=True)["Survived"].mean().reset_index()
summary.to_csv("titanic_summary.csv", index=False)
df.to_csv("titanic_cleaned.csv", index=False)
print(pd.read_csv("titanic_summary.csv"))
