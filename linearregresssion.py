#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
#Put the data
X=np.array([12,90,78,45,88])
Y=np.array([12,34,65,78,98])
X=X.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
print("\n Slope(m):",round(model.coef_[0],2))
print("\n Intercept(b):",round(model.intercept_,2))
y_pred=model.predict(X_test)
print("\n Actual Test:",y_test)
print("\n Predicted Score:",y_pred.round(3))
print("\n R^2 Score:",round(r2_score(y_test,y_pred),4))
plt.figure(figsize=(8,6))
plt.scatter(X,Y,color='blue',label='Actual Data')
#Sorted for better visualization
X_sorted = np.sort(X, axis=0)
#Plotting the graph to plote that opens with the help of graph.py 
plt.plot(X_sorted,model.predict(X_sorted),color='Red',label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Regression_Line: Study Hours Vs Exam Marks")
plt.legend()
plt.show()
