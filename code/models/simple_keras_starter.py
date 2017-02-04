from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

keras_simple_model = Sequential()
keras_simple_model.add(Dense(3000, input_dim = 2304, init='normal', activation = 'relu'))#Input layer
keras_simple_model.add(Dense(2000, init='normal', activation = 'relu'))#Hidden layer
keras_simple_model.add(Dense(7, init='normal', activation='sigmoid'))#Output Layer
keras_simple_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])# Compile model
keras_simple_model.fit(x_train_vectoral, y_train, validation_data=(x_public_test_vectoral, y_public_test), nb_epoch=10, batch_size=5000, verbose=1)#Fit
score = keras_simple_model.evaluate(x_train_vectoral, y_train, verbose=0)#Get results
print(score[1]*100)#Train result