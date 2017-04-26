from pre_trained_models.VG16 import vg_16_get_features
from pre_trained_models.VG19 import vg_19_get_features
from pre_trained_models.inception import inception_get_features
from format_data import get_data_in_matrix_format
import pandas as pd


N_TEST = 28708
raw_data_csv_file_name = '../../data/fer2013.csv'


def run_merge_model():

    raw_data = pd.read_csv(raw_data_csv_file_name)
    emotion = raw_data[['emotion']]
    pixels = raw_data[['pixels']]
    # Get data in matrix form
    (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
     y_train, y_public_test, y_private_test) = \
        get_data_in_matrix_format(emotion, pixels)
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
