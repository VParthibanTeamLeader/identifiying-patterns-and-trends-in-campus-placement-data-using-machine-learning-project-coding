import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score
import warnings

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers

from matplotlib.offsetbox import martist

from sklearn.svm import SVC
from sklearn import svm


warnings.filterwarnings('ignore')

df = pd.read_csv(r"/content/collegePlace.csv")
df.head()

df.shape

df.info()

df.isnull().sum()


def transformationplot(feature):
  plt.figure(figsize=(12,5))
  plt.subplot(1,2,1)
  sns.distplot(feature)
transformationplot(np.log(df['Age']))

df = df.replace(['Male'],[0])
df = df.replace(['Female'],[1])

df = df.replace(['Computer Science'],[0])
df = df.replace(['Information Technology'],[1])
df = df.replace(['Electronics And Communication'],[2])
df = df.replace(['Mechanical'],[3])
df = df.replace(['Electrical'],[4])
df = df.replace(['Civil'],[5])
df

df.info()



plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(df['CGPA'],color='r')

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(df['PlacedOrNot'],color='g')


plt.figure(figsize=(30,5))
plt.subplot(1,4,1)
sns.countplot(x="PlacedOrNot",data=df, ec='black')
plt.subplot(1,4,2)
sns.countplot(y="Stream",data=df, ec='black') 
plt.show()



plt.figure(figsize=(20,5))
plt.subplot(131)
sns.countplot(x='PlacedOrNot', data=df, hue='CGPA', ec='black')

sns.swarmplot(x='PlacedOrNot',y='CGPA', hue='Stream', data=df)

df.describe()


x = df.drop('PlacedOrNot',axis=1)
y=df['PlacedOrNot']

x

y

sc = StandardScaler()
x = sc.fit_transform(x)
x = pd.DataFrame(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.11, stratify=y, random_state=42)


print(x_train.shape)
print(x_train.shape)



svm = SVC()
svm.fit(x_train,y_train)
SVC()


classifier = svm.SVC()
x_test = np.array(x_test, dtype = float)
y_test = np.array(y_test, dtype = float)
classifier.fit(x_train, y_train)
SVC()

x_test_prediction = classifier.predict(x_test)
y_pred= accuracy_score(x_test_prediction,y_test)
y_pred

best_k = {"Regular":0}
best_score = {"Regular":0}
for k in range(3, 50, 2): 
  knn_temp = KNeighborsClassifier(n_neighbors=k)
  knn_temp.fit(x_train, y_train)
  knn_temp_pred = knn_temp.predict(x_test)
  score = metrics.accuracy_score(y_test, knn_temp_pred) * 100
  if score >=  best_score["Regular"]and score < 100:
    best_score["Regular"] = score
    best_k["Regualar"] = k



print("---Results---\nk: {}\nScore: {}".format(best_k, best_score))
knn = KNeighborsClassifier(n_neighbors=best_k["Regualar"])
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
testd = accuracy_score(knn_pred, y_test)





classifier = Sequential()

#add input layer and first hidden layer
classifier.add(keras.layers.Dense(6,activation = 'relu',input_dim = 6))
classifier.add(keras.layers.Dropout(0.50))

#add second hidden layer
classifier.add(keras.layers.Dense(6,activation = 'relu'))
classifier.add(keras.layers.Dropout(0,50))

#final or output layer
classifier.add(keras.layers.Dense(1,activation = 'sigmoid'))

#compiling the model
loss_1 = tf. keras.losses.BinaryCrossentropy()
classifier.compile(optimizer = 'Adam', loss= loss_1, metrics = ['accuracy'])

#fitting th model
classifier.fit(x_train, y_train, batch_size = 20, epochs = 100)


import pickle 
pickle.dump(knn,open("placement.pkl",'wb'))
model = pickle.load(open('placement.pkl','rb'))

input_data = [[22,0,2,1,8,1]]

prediction = knn.predict(input_data)
print(prediction)
if (prediction[0]==0):
  print('not placed')
else:
      print('placed')
