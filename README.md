# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.

2.Import required libraries.

3.Load the car price dataset.

4.Select input features and target variable.

5.Split data into training and testing sets.

6.Train the Linear Regression model.

7.Train the Polynomial Regression model.

8.Predict prices and evaluate performance.

9.Display results and stop the program.
## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Johan Renish A
RegisterNumber:  212225040159
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import matplotlib.pyplot as plt
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_linear= lr.predict(X_test)
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

print("="*50)
print('Name: Johan Renish A ')
print('Reg. No: 212225040159')
print("="*50)
print("Linear Regression: ")
mse=mean_squared_error(y_test,y_pred_linear)
print(f"MSE: {mean_squared_error(y_test,y_pred_linear):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_linear):.2f}")
r2score=r2_score(y_test,y_pred_linear)
print(f'R2 Score: {r2score:.2f}')
print("="*50)
print("Polynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"MSE: {mean_absolute_error(y_test, y_pred_poly):.2f}")
print(f"R²: {r2_score(y_test, y_pred_poly):.2f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
*/
```
## Output:
<img width="1363" height="759" alt="1" src="https://github.com/user-attachments/assets/26f6fda2-70d7-4bc1-8f83-5ed62e8be8a2" />
<img width="1241" height="603" alt="2" src="https://github.com/user-attachments/assets/52af7437-7b14-494b-88e5-61ba9e010c5b" />
## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
