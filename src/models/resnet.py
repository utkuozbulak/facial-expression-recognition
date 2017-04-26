from keras.applications import ResNet50
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD, Adamax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, InputLayer
from format_data import get_data_in_matrix_format
from keras.layers import Dense
from keras.constraints import maxnorm
from keras import backend as K

IMAGE_SHAPE = (48, 48, 1)
raw_data_csv_file_name = '../../data/fer2013.csv'


if __name__ == "__main__":

    N_TEST = 25000

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


    resnetPreTrained = ResNet50(include_top=False, input_shape=(200,200,3))

    x_train_feature_map = np.empty([N_TEST, 2048])

    for i in range(5):
        print("loading images {}".format(i*5000))
        load_range = (i*5000)
        f_l = np.empty([int(N_TEST/20), 200, 200, 3])
        for index, item in enumerate(f_l[load_range:load_range+4999]):  # Refill the list
            for index, item in enumerate(f_l):  # Refill the list
                for d in range(3):
                    item[0:48, 0:48, d] = x_train_matrix[index]


        picture_train_features = resnetPreTrained.predict(f_l)
        del(f_l)

        #BUILD NEW TRAIN FEATURE INPUT
        for idx_pic, picture in enumerate(picture_train_features):
            idx = idx_pic + (i*5000)
            x_train_feature_map[idx] = picture[0][0]

    f_t = np.empty([3588, 200, 200, 3])
    for index, item in enumerate(f_t):  # Refill the list
        for d in range(3):
            item[0:48, 0:48,d] = x_public_test_matrix[index]


    picture_test_features = resnetPreTrained.predict(f_t)
    del(f_t)

    #BUILD NEW TEST
    x_test_feature_map  = np.empty([3588, 2048])
    for idx_pic, picture in enumerate(picture_test_features):
        x_test_feature_map[idx_pic] = picture[0][0]

    print("here")
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Dense(1024, input_shape=(2048,),activation='relu'))
        model.add(Dense(512, input_shape=(1024,),activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        adamax = Adamax()
        model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])
        model.fit(x_train_feature_map, y_train[0:N_TEST], validation_data=(x_train_feature_map, y_train[0:N_TEST]), nb_epoch=10, batch_size=64)

    score = model.evaluate(x_test_feature_map, y_public_test, batch_size=64)
    print("Result")
    print(score)
        #
    # model.add(Flatten())
    # model.add(Dense(1000, activation='relu', W_constraint=maxnorm(3)))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    #
    # epochs = 10
    # lrate = 0.02
    # decay = lrate / epochs
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # adamax = Adamax()
    # model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])
    # model.fit(f_l, y_train, validation_data=(f_l, y_train), nb_epoch=10, batch_size=32)
    #
    # print("HERE")
