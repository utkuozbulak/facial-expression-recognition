from pre_trained_models.VG16 import vg_16_get_features
from pre_trained_models.VG19 import vg_19_get_features
from pre_trained_models.inception import inception_get_features
from format_data import get_data_in_matrix_format
import pandas as pd
from keras.optimizers import Adamax
from keras.models import Sequential
from keras.layers import Dense, Dropout, Merge, Flatten

N_TEST = 28708
raw_data_csv_file_name = '../../data/fer2013.csv'


def run_merge_model():
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

    vg_16_train_features, vg_16_test_features = vg_16_get_features()
    print("loaded vg16")
    vg_19_train_features, vg_19_test_features = vg_19_get_features()
    print("loaded vg19")
    inception_train_features, inception_test_features = inception_get_features()
    print("loaded inception")

    model_vg16 = Sequential()
    model_vg16.add(Dense(256, input_shape=(512,), activation='relu'))
    model_vg16.add(Dense(128, input_shape=(256,), activation='softmax'))
    model_vg16.add(Dropout(0.5))

    model_vg19 = Sequential()
    model_vg19.add(Dense(256, input_shape=(512,), activation='relu'))
    model_vg19.add(Dense(128, input_shape=(256,), activation='softmax'))
    model_vg19.add(Dropout(0.5))

    model_inception = Sequential()
    model_inception.add(Dense(512, input_shape=(2048,), activation='relu'))
    model_inception.add(Dense(128, input_shape=(512,), activation='softmax'))
    model_inception.add(Dropout(0.5))

    model_merge = Sequential()
    model_merge.add(Merge([model_vg16, model_vg19], mode='concat'))
    model_merge.add(Dense(1000, input_shape=(2000,), activation='relu'))
    model_merge.add(Dense(100, input_shape=(1000,), activation='relu'))

    model_merge.add(Dense(num_classes, activation='softmax'))

    adamax = Adamax()
    model_merge.compile(loss='categorical_crossentropy', optimizer=adamax,
                        metrics=['accuracy'])
    model_merge.fit([vg_16_train_features, vg_19_train_features], y_train,
                    validation_data=(
                    [[vg_16_train_features, vg_19_train_features], y_train]),
                    epochs=80, batch_size=512)
    score = model_merge.evaluate([vg_16_test_features, vg_19_test_features],
                                 y_public_test, batch_size=512)

    print("Result")
    print(score)


if __name__ == "__main__":
    run_merge_model()
