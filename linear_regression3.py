#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
#Make Data
X=np.array([12,13,14,19])
Y=np.array([10,14,190,192])
#sklearn takes array in 2d array
X=X.reshape(-1,1)
#make testing and training
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=23)
model=LinearRegression()
model.fit(X_train,y_train)
print("\n Slope(m):",round(model.coef_[0],2))
print("\n Intercept(b):",round(model.intercept_,2))
y_pred=model.predict(X_test)
print("\n Actual Score:",y_test)
print("\n Predicted Score:",y_pred.round(2))
#Checking the accuracy
print("R2_Score:",round(r2_score(y_test,y_pred),4))
#plotting the graph
plt.figure(figsize=(6,4))
plt.scatter(X,Y,color='Blue',label='Actual Data')
plt.plot(X,model.predict(X),color='Brown',label='Regression Line')
plt.xlabel("Study Marks")
plt.ylabel("Exam Marks")
plt.title("Figuring out the prediction")
plt.legend()
plt.show()
