import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

data = pd.read_csv('housing.csv')

data.dropna(inplace=True)
ocean_data = pd.get_dummies(data.ocean_proximity).astype(int)
data = data.join(ocean_data).drop("ocean_proximity", axis=1)

X = data.drop("median_house_value",axis=1)
Y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred)
r2_linear = r2_score(y_test, y_pred)

clf = GradientBoostingRegressor(
    max_depth=6,
    learning_rate = 0.1,
    n_estimators = 100,
    subsample = 0.8,
    random_state = 42
)

clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)

mse_clf = mean_squared_error(y_test, y_pred_clf)
r2_clf = r2_score(y_test,y_pred_clf)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: True VS Predicted")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color = 'red', linestyle='--')
plt.grid()

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_clf, color='green', alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Gradient Boosting: True VS Predicted")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color = 'red', linestyle='--')
plt.grid()

plt.tight_layout()
plt.show()

print(f"Linear Regression MSE: {mse_linear:.2f}, r2: {r2_linear:.2f}")
print(f"Gradient Boosting MSE: {mse_clf:.2f}, r2: {r2_clf:.2f}")