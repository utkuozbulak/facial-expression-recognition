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

    x_train_matrix = x_train_matrix - x_train_matrix.mean(axis=0)
    x_public_test_matrix = x_public_test_matrix - x_public_test_matrix.mean(axis=0)
    # 7 Classes
    num_classes = y_train.shape[1]

    vg_16_train_features, vg_16_test_features, vg_16_model\
        = vg_16_get_features(True)
    print("loaded vg16")
    vg_19_train_features, vg_19_test_features, vg_19_model \
        = vg_19_get_features(True)
    print("loaded vg19")
    inception_train_features, inception_test_features, inception_model \
        = inception_get_features(True)
    print("loaded inception")

    vg16_prob = vg_16_model.predict_proba(vg_16_test_features)
    vg19_prop = vg_19_model.predict_proba(vg_19_test_features)
    # inception_prob = inception_model.predict_proba(inception_test_features)

    combined_probability = (vg16_prob + vg19_prop)/2

    number_correct = 0
    results = []
    for x in range(len(x_public_test_matrix)):
        chosen_class = list(combined_probability[x]).index(max(combined_probability[x]))
        actual_class = list(y_public_test[x]).index(1)
        if chosen_class == actual_class:
            number_correct += 1

        chosen_prob = combined_probability[x][chosen_class]
        correct_prob = combined_probability[x][actual_class]

        results.append([chosen_class == actual_class,
                        chosen_class, actual_class, chosen_prob, correct_prob])

    for res in results:
        print("correct {} , predicted class {}, actual class {}, "
              "predicted prob {}, prob of correct class {}".format(
                res[0], res[1], res[2], res[3], res[4]))

    print(number_correct/len(y_public_test))

if __name__ == "__main__":
    run_merge_model()
