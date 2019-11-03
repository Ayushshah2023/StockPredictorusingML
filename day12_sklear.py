import sklearn
'''import json
import requests
from requests_oauthlib import 0Auth1
api_key=input("Enter the api keys")
api_secret_key=input("Enter the 
keyword'''
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
data=pd.read_csv('infy.csv')
#print(data)
data=data[['Open Price','Low Price','High Price','Close Price',"Average Price"]]
#print(data)
#In o1,h2,l2,c1 its label will be c4 as we have to predict he price of 3 days in the future
data['label']=data['Close Price'].shift(-3)
print(data)
#X will be represented as features
#y willl represent as label
X=np.array(data.drop(['label'],axis=1))
X_lately=X[-3:]
X=X[:-3]
#print(X)
y=np.array(data['label'])
y=y[:-3]
print(X.shape,y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
confidence=clf.score(X_test,y_test)
print(confidence," is the confidence score")
forecast_set=clf.predict(X_lately)
print(forecast_set)
