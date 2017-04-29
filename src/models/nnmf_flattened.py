from format_data import get_data_in_matrix_format
import pandas as pd
import numpy as np
from keras.optimizers import Adamax
from keras.models import Sequential
from keras.layers import Dense, Dropout, Merge
from pre_processing.nnmf.nnmf_feature_extraction import get_nnmf_features

N_TEST = 28708
COMMON_DROPOUT_RATE = 0.2
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

    x_train_matrix = x_train_matrix / 255.0
    x_public_test_matrix = x_public_test_matrix / 255.0
    # Put values between 1 and 0
    # x_train_matrix = x_train_matrix / 255.0
    # x_public_test_matrix = x_public_test_matrix / 255.0
    # 7 Classes
    num_classes = y_train.shape[1]

    print("Loading x_train features")
    x_train_features = get_nnmf_features(x_train_matrix)
    x_train_flattern = np.zeros((len(x_train_matrix), 288))
    for idx, train_row in enumerate(x_train_features):
        x_train_flattern[idx, :] = train_row.flatten()

    print("Loading x_test features")
    x_test_features = get_nnmf_features(x_public_test_matrix)
    x_test_flattern = np.zeros((len(x_test_features), 288))
    for idx, train_row in enumerate(x_test_features):
        x_test_flattern[idx, :] = train_row.flatten()

    model = Sequential()
    model.add(Dense(144, input_shape=(288,), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(78, input_shape=(144,), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(39, input_shape=(78,)))
    model.add(Dense(num_classes, activation='softmax'))
    adamax = Adamax()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adamax, metrics=['accuracy'])
    model.fit(x_train_flattern, y_train,
              validation_data=(x_train_flattern, y_train),
              nb_epoch=100, batch_size=128)

    score = model.evaluate(x_test_flattern,
                           y_public_test, batch_size=128)
    print("Result")
    print(score)





if __name__ == "__main__":
    run_merge_model()
