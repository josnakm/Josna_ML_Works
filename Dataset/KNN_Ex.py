#KNN Algorithm
#importing algorithm

import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

#importing datasets
data_set = pd.read_csv('D:\\DataScience\\heart.csv')
df=pd.DataFrame(data_set)
print(df.to_string())

#Extracting independent and dependend variables
x= data_set.iloc[:, [2,3]].values
y= data_set.iloc[:, 4].values
df2=pd.DataFrame(x)
print(df2.to_string())

#Spliting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

print("x_train b4 scaling..")
df3=pd.DataFrame(x_train)
print(df3.to_string())

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
print("x_train after scaling...")
df4=pd.DataFrame(x_train)
print(df4.to_string())

#Fitting KNN classifier to the training test
from sklearn.neighbors import KNeighborsClassifier

classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred= classifier.predict(x_test)
print(y_pred)

print("Prediction comparison")
ddf=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
print(ddf.to_string())

#Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
