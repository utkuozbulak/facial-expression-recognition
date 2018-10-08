from keras.applications import VGG16
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import Adamax
from keras.models import Sequential
from keras.layers import Dense, Dropout
from format_data import get_data_in_matrix_format
from keras.layers.advanced_activations import PReLU
import os.path

N_TEST = 28708
raw_data_csv_file_name = '../../data/fer2013.csv'


def vg_16_get_features(run_evaluation_model=False):

    def run_model():
        print("here")
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Dense(256, input_shape=(512,), activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(128, input_shape=(256,), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, input_shape=(64,)))
            model.add(Dense(num_classes, activation='softmax'))
            adamax = Adamax()
            model.compile(loss='categorical_crossentropy',
                          optimizer=adamax, metrics=['accuracy'])
            model.fit(x_train_feature_map, y_train,
                      validation_data=(x_train_feature_map, y_train),
                      nb_epoch=100, batch_size=128)

        score = model.evaluate(x_test_feature_map,
                               y_public_test, batch_size=128)
        print("Result")
        print(score)
        return model

    raw_data = pd.read_csv(raw_data_csv_file_name)
    emotion = raw_data[['emotion']]
    pixels = raw_data[['pixels']]
    # Get data in matrix form
    (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
     y_train, y_public_test, y_private_test) = \
        get_data_in_matrix_format(emotion, pixels)
    # Only used train and public test for now
    x_train_matrix = x_train_matrix.astype('float32')
    x_train_matrix = x_train_matrix - x_train_matrix.mean(axis=0)

    x_public_test_matrix = x_public_test_matrix.astype('float32')
    x_public_test_matrix = x_public_test_matrix - x_public_test_matrix.mean(axis=0)

    # Put values between 1 and 0

    # 7 Classes
    num_classes = y_train.shape[1]

    if os.path.exists("./pre_saved_features/vg16trainfeatures.npy"):
        x_train_feature_map = \
            np.load("./pre_saved_features/vg16trainfeatures.npy")
        x_test_feature_map = \
            np.load("./pre_saved_features/vg16testfeatures.npy")
    else:

        vg = VGG16(include_top=False, input_shape=(48, 48, 3),
                   pooling='avg', weights='imagenet')

        x_train_feature_map = np.empty([N_TEST, 512])
        f_l = np.empty([int(N_TEST), 48, 48, 3])
        for index, item in enumerate(f_l):
            # im = Image.fromarray(x_train_matrix[index])
            # rotate90 = im.rotate(90)
            # flipimage = im.transpose(Image.FLIP_LEFT_RIGHT)

            item[:, :, 0] = x_train_matrix[index]
            item[:, :, 1] = x_train_matrix[index]
            item[:, :, 2] = x_train_matrix[index]

        picture_train_features = vg.predict(f_l)
        del (f_l)

        for idx_pic, picture in enumerate(picture_train_features):
            x_train_feature_map[idx_pic] = picture

        f_t = np.empty([3588, 48, 48, 3])
        for index, item in enumerate(f_t):  # Refill the list
            # im = Image.fromarray(x_public_test_matrix[index])
            # rotate90 = im.rotate(90)
            # flipimage = im.transpose(Image.FLIP_LEFT_RIGHT)
            item[:, :, 0] = x_public_test_matrix[index]
            item[:, :, 1] = x_public_test_matrix[index]
            item[:, :, 2] = x_public_test_matrix[index]

        picture_test_features = vg.predict(f_t)
        del (f_t)

        # BUILD NEW TEST
        x_test_feature_map = np.empty([3588, 512])
        for idx_pic, picture in enumerate(picture_test_features):
            x_test_feature_map[idx_pic] = picture

        np.save("./pre_saved_features/vg16testfeatures", x_test_feature_map)
        np.save("./pre_saved_features/vg16trainfeatures", x_train_feature_map)

    if run_evaluation_model:
        model = run_model()
    else:
        model = None

    return x_train_feature_map, x_test_feature_map, model


if __name__ == "__main__":
    vg_16_get_features(True)
