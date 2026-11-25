import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")
print(df.head())

features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("Intercept:", model.intercept_)
for name, coef in zip(features, model.coef_):
    print(name, "coefficient:", coef)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

new_house = pd.DataFrame([[6000, 3, 2, 2, 1]], columns=features)
print("Predicted price for a new house:", model.predict(new_house)[0])
