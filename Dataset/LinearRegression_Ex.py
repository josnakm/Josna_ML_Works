#Simple linear regression
#importing libraries

import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

#Load data
data_set= pd.read_csv('D:\DataScience\Salary_Data(1).csv')
print(data_set.describe())

#print data
print("Dataset")
df=pd.DataFrame(data_set)
print(df.to_string())

#Pick colomns for x and y
X= data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values

print("X=\n",X)
df2=pd.DataFrame(X)
print("X Data-Experience")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y Data-Salary")
print(df3.to_string())
print("Y array\n")
print(y)

#Load dataset scling module
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,
y, test_size= .2, random_state=0)

#Load liniear regression clss
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()

#Fitting the simple linear regresison model to the training test
regressor.fit(x_train, y_train)

x_pred= regressor.predict(x_train)
print("Prediction result on Test Data")
y_pred = regressor.predict(x_test)
dfs=pd.DataFrame(x_test)
print("X-test")
print(dfs)
df2 = pd.DataFrame({'Actual Y-Data': y_test,'Predicted Y-Data': y_pred})
print(df2.to_string())

print("Mean")
print(df['Salary'].mean())
from sklearn import metrics

print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')
y_pred2 = regressor.predict([[3.5]])
print("exp 3.5")
print(y_pred2)
arr=np.array([[5.6],[8.0]])
print("arr \n")
print(arr)
y_pred3 = regressor.predict(arr)
print(y_pred3)

