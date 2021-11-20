import pickle
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

NAME = f'cat-and-dog-prediction-{int(time.time())}'

tensorboard = TensorBoard(log_dir = f'logs\\{NAME}\\')

X = X/255

model = Sequential()

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape =X.shape[1:], activation='relu' ))

model.add(Dense(128, input_shape =X.shape[1:], activation='relu' ))

model.add(Dense(128, input_shape =X.shape[1:], activation='relu' ))

model.add(Dense(2, activation= 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X, y, epochs=10, validation_split = 0.1, batch_size=16 ,callbacks=tensorboard)

model.save('model.h5', save_format="h5")
