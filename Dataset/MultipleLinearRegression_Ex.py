#Multiple linear regression
#importing libraries

import pandas as pd
import numpy as np

#Importing dataset
data_set= pd.read_csv('D:\DataScience\Stores.csv')
print(data_set.to_string())

#Extracting independent and dependent variables
x= data_set.iloc[:, :-1].values
y= data_set.iloc[:, 4].values

df2=pd.DataFrame(x)
print("X=")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y=")
print(df3.to_string())

#Catogorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x= LabelEncoder()

x[:, 3]= labelencoder_x.fit_transform(x[:,3])
dt=pd.DataFrame(x)
print()
print(dt.to_string())
print()

#State colomn
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder =
'passthrough')
x = ct.fit_transform(x)

#avoiding dummy variable trap
x = x[:, 1:]
df4=pd.DataFrame(x)
print("Updated X=")
print(df4.to_string())

#Spliting the dataset into training set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)

#Fitting the MLR into training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set result
y_pred= regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())

print("Mean")
print(data_set.describe())
print()

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
#Predicting the accuracy score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")