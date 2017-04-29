from keras.optimizers import SGD, Adamax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, InputLayer
from format_data import get_data_in_matrix_format
from keras.constraints import maxnorm
from keras import backend as K
import pandas as pd
import numpy as np
import tensorflow as tf

K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

IMAGE_SHAPE = (48, 48, 1)
raw_data_csv_file_name = '../data/fer2013.csv'


if __name__ == "__main__":

    raw_data = pd.read_csv(raw_data_csv_file_name)
    emotion = raw_data[['emotion']]
    pixels = raw_data[['pixels']]
    # Get data in matrix form
    (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
     y_train, y_public_test, y_private_test) = get_data_in_matrix_format(emotion, pixels)
    # Only used train and public test for now
    x_train_matrix = x_train_matrix.astype('float32')
    x_public_test_matrix = x_public_test_matrix.astype('float32')
    # Put values between 1 and 0
    x_train_matrix = x_train_matrix / 255.0
    x_public_test_matrix = x_public_test_matrix / 255.0
    # 7 Classes
    num_classes = y_train.shape[1]
    # Recreate array, this was necessary because the original format was:
    # [28708,48,48], but Keras needs an indicator on depth so, added 1 as depth
    f_l = np.empty([28708,1,48,48])

    for index,item in enumerate(f_l):  # Refill the list
        item[0] = x_train_matrix[index]

    f_t = np.empty([3588,1,48,48])
    for index,item in enumerate(f_t):  # Refill the list
        item[0] = x_public_test_matrix[index]

    print("making model")
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Convolution2D(48, 4, 4, input_shape=(1, 48, 48), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(48, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(48, 2, 2, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        epochs = 10
        lrate = 0.02
        decay = lrate/epochs
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        adamax = Adamax()
        model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])
        model.fit(f_l, y_train, validation_data=(f_l, y_train), nb_epoch=10, batch_size=32)

    score = model.evaluate(f_t, y_public_test, batch_size=32)
    print("Result")
    print(score)