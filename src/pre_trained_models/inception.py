from keras.applications import Xception
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import Adamax
from keras.models import Sequential
from keras.layers import Dense, Dropout
from format_data import get_data_in_matrix_format
from PIL import Image
import os

raw_data_csv_file_name = '../../data/fer2013.csv'
N_TEST = 28708


def inception_get_features(run_evaluation_model=False):
    def run_model():
        print("here")
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Dense(1024, input_shape=(2048,), activation='relu'))
            model.add(Dense(512, input_shape=(1024,), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, input_shape=(64,)))
            model.add(Dense(num_classes, activation='softmax'))
            adamax = Adamax()
            model.compile(loss='categorical_crossentropy', optimizer=adamax,
                          metrics=['accuracy'])
            model.fit(x_train_feature_map, y_train,
                      validation_data=(x_train_feature_map, y_train),
                      nb_epoch=70, batch_size=64)

        score = model.evaluate(x_test_feature_map, y_public_test, batch_size=64)
        return model

    raw_data = pd.read_csv(raw_data_csv_file_name)
    emotion = raw_data[['emotion']]
    pixels = raw_data[['pixels']]
    # Get data in matrix form
    (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
     y_train, y_public_test, y_private_test) = get_data_in_matrix_format(
        emotion, pixels)
    # Only used train and public test for now
    x_train_matrix = x_train_matrix.astype('float32')
    x_public_test_matrix = x_public_test_matrix.astype('float32')
    # Put values between 1 and 0
    x_train_matrix = x_train_matrix / 255.0
    x_public_test_matrix = x_public_test_matrix / 255.0
    # 7 Classes
    num_classes = y_train.shape[1]

    if os.path.exists("./pre_saved_features/inceptiontrainfeatures.npy"):
        x_train_feature_map = np.load(
            "./pre_saved_features/inceptiontrainfeatures.npy")
        x_test_feature_map = np.load(
            "./pre_saved_features/inceptiontestfeatures.npy")
    else:

        xception_pre_trained = Xception(include_top=False,
                                        input_shape=(96, 96, 3), pooling='avg',
                                        weights='imagenet')

        x_train_feature_map = np.empty([N_TEST, 2048])

        f_l = np.empty([int(N_TEST), 48 * 2, 48 * 2, 3])
        for index, item in enumerate(f_l):  # Refill the list
            im = Image.fromarray(x_train_matrix[index])
            resized_image = im.resize((48 * 2, 48 * 2), Image.NEAREST)
            arr = np.reshape(
                np.fromiter(iter(resized_image.getdata()), np.uint8), (96, 96))
            for d in range(3):
                item[:, :, d] = arr

        picture_train_features = xception_pre_trained.predict(f_l)
        del f_l

        # BUILD NEW TRAIN FEATURE INPUT
        for idx_pic, picture in enumerate(picture_train_features):
            x_train_feature_map[idx_pic] = picture

        print("converting test features")
        f_t = np.empty([3588, 96, 96, 3])
        for index, item in enumerate(f_t):  # Refill the list
            im = Image.fromarray(x_public_test_matrix[index])
            resized_image = im.resize((48 * 2, 48 * 2), Image.NEAREST)
            arr = np.reshape(
                np.fromiter(iter(resized_image.getdata()), np.uint8), (96, 96))
            for d in range(3):
                item[:, :, d] = arr

        picture_test_features = xception_pre_trained.predict(f_t)
        del f_t

        # BUILD NEW TEST
        x_test_feature_map = np.empty([3588, 2048])
        for idx_pic, picture in enumerate(picture_test_features):
            x_test_feature_map[idx_pic] = picture

        np.save("./pre_saved_features/inceptiontestfeatures",
                x_test_feature_map)
        np.save("./pre_saved_features/inceptiontrainfeatures",
                x_train_feature_map)

    if run_evaluation_model:
        model = run_model()
    else:
        model = None
    return x_train_feature_map, x_test_feature_map, model


if __name__ == "__main__":
    inception_get_features(True)
