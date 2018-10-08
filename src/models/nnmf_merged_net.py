from format_data import get_data_in_matrix_format
import pandas as pd
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
    print("Loading x_test features")
    x_test_features = get_nnmf_features(x_public_test_matrix)
    
    model_feature1 = Sequential()
    model_feature1.add(Dense(24, input_shape=(48,), activation='relu'))
    model_feature1.add(Dense(24, input_shape=(24,), activation='relu'))
    model_feature1.add(Dropout(COMMON_DROPOUT_RATE))
    
    model_feature2 = Sequential()
    model_feature2.add(Dense(24, input_shape=(48,), activation='relu'))
    model_feature2.add(Dense(24, input_shape=(24,), activation='relu'))
    model_feature2.add(Dropout(COMMON_DROPOUT_RATE))
    
    model_feature3 = Sequential()
    model_feature3.add(Dense(24, input_shape=(48,), activation='relu'))
    model_feature3.add(Dense(24, input_shape=(24,), activation='relu'))
    model_feature3.add(Dropout(COMMON_DROPOUT_RATE))
   
    model_feature4 = Sequential()
    model_feature4.add(Dense(24, input_shape=(48,), activation='relu'))
    model_feature4.add(Dense(24, input_shape=(24,), activation='relu'))
    model_feature4.add(Dropout(COMMON_DROPOUT_RATE))

    model_feature5 = Sequential()
    model_feature5.add(Dense(24, input_shape=(48,), activation='relu'))
    model_feature5.add(Dense(24, input_shape=(24,), activation='relu'))
    model_feature5.add(Dropout(COMMON_DROPOUT_RATE))

    model_feature6 = Sequential()
    model_feature6.add(Dense(24, input_shape=(48,), activation='relu'))
    model_feature6.add(Dense(24, input_shape=(24,), activation='relu'))
    model_feature6.add(Dropout(COMMON_DROPOUT_RATE))

    model_merge = Sequential()
    model_merge.add(Merge([model_feature1, model_feature2, model_feature3,
                           model_feature4, model_feature5, model_feature6],
                          mode='concat'))
    model_merge.add(Dense(60, input_shape=(120,), activation='relu'))
    model_merge.add(Dense(num_classes, activation='softmax'))

    adamax = Adamax()
    model_merge.compile(loss='categorical_crossentropy', optimizer=adamax,
                        metrics=['accuracy'])
    model_merge.fit([x_train_features[:, :, 0], x_train_features[:, :, 1],
                     x_train_features[:, :, 2], x_train_features[:, :, 3],
                     x_train_features[:, :, 4], x_train_features[:, :, 5]],
                    y_train,
                    validation_data=(
                    [x_train_features[:, :, 0], x_train_features[:, :, 1],
                     x_train_features[:, :, 2], x_train_features[:, :, 3],
                     x_train_features[:, :, 4], x_train_features[:, :, 5]],
                    y_train),
                    epochs=250, batch_size=512)

    score = model_merge.evaluate([x_test_features[:, :, 0], x_test_features[:, :, 1],
                                  x_test_features[:, :, 2], x_test_features[:, :, 3],
                                  x_test_features[:, :, 4], x_test_features[:, :, 5]],
                                 y_public_test, batch_size=512)
    print("Result")
    print(score)

if __name__ == "__main__":
    run_merge_model()
