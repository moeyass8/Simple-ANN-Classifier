import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('/home/moe/Desktop/AI/AI in Cyber Security/Finalproject/heart.csv')
x = dataset.iloc[:, 0:13]
y = dataset.iloc[:, 13]
#labelencoder1 = LabelEncoder()
#x1 = x.iloc[:, 0]
#x1[:, 0] = labelencoder1.fit_transform(x1[:, 0])


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


sc = StandardScaler()
x_tr = sc.fit_transform(x_train)
x_tst = sc.fit_transform(x_test)

#Making the ANN
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_tr, y_train, batch_size = 10, epochs = 100)

y_predict = classifier.predict(x_tst)
y_predict = (y_predict > 0.5)

newdata = pd.read_csv('/home/moe/Desktop/AI/AI in Cyber Security/Finalproject/new.csv')
newdata1 = sc.fit_transform(newdata)
predict = classifier.predict(newdata1)


cm = confusion_matrix(y_test, y_predict)
