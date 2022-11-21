
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np

#Load Data
data=np.load('data.npy')
target=np.load('target.npy')

#Generate Model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(96,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))

model.add(Dense(2,activation='softmax'))

#Compile our model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Split out data for traing and testing
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.3)

#train our model for 10 epochs
history=model.fit(train_data,train_target,epochs=10,validation_split=0.3)

#Save trained model to use in future
model.save("Card_Detector.model")


