"""
Regression ✅

Do variables ke darmiyan relationship find karna

Linear Regression ✅

Us relationship ki basis pe future values predict karna

Formula y = mx + b ✅
PartNaamKaamyDependent variableJo predict hota haixIndependent variableJo hum dete hainmSlopeRelationship ki steepnessbInterceptJab x=0 ho toh y ki value

Ek real example tumhare words mein:

"Agar main study hours (x) ki basis pe exam score (y) predict karna chahoon — toh linear regression slope (m) se batayega ke har 1 extra hour padhne pe score kitna badhta hai."
1. MSE — model kitna galat hai measure karta hai. Jitna kam MSE utna acha model.
2. R² score — 0 se 1 tak. 1 matlab perfect prediction
#Actual Flow
 Data banao
    ↓
Train/Test split karo
    ↓
Model banao (LinearRegression)
    ↓
Model ko train karo (fit)
    ↓
Prediction karo (predict)
    ↓
Accuracy check karo (R², MSE)
    ↓
Graph banao (matplotlib) """
#Import first libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
#Make Data

# X = study hours (input)
# y = exam scores (output)
#Both X and Y have same size 
X= np.array([1,2,3,4,5,6,7,8,9,10])
Y=np.array([10,20,30,40,50,60,70,80,90,100])
#Divide the data into two steps test and training 80%training 20% testing
X=X.reshape(-1,1)#sklearn ko 2 d array chaye
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#make model and train it
model=LinearRegression()
model.fit(X_train,y_train)


print("Slope (m):", round(model.coef_[0], 2))

print("Intercept (b):", round(model.intercept_, 2))

#Make Prediction
y_pred=model.predict(X_test)
print("\n Actual Score:",y_test)
print("\n Predicted Score:",y_pred.round(2))
#Check accuracy
print("\n R^2 Score:",round(r2_score(y_test,y_pred),4))
#Make Graph
plt.figure(figsize=(8,5))
plt.scatter(X,Y,color='blue',label='Actual data')
plt.plot(X,model.predict(X),color='red',label='Regression line')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title('Linear Regression- Study Hours vs Exam Score')
plt.legend()
plt.show()