import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = np.array([12,32,12,213,90]).reshape(-1,1)
Y = np.array([10,90,45,89,23])

X_train, X_test, Y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

print("Slope (m):", round(model.coef_[0], 2))
print("Intercept (b):", round(model.intercept_, 2))

y_pred = model.predict(X_test)

print("\nActual Score:", y_test)
print("\nPredicted Score:", y_pred.round(2))
print("\nR² Score:", round(r2_score(y_test, y_pred), 4))

plt.figure(figsize=(8,5))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='pink', label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Linear Regression - Study Hours vs Exam Marks")
plt.legend()
plt.show()