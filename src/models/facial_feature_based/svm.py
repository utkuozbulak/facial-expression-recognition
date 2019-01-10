import multiprocessing
from functools import partial
from itertools import product

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import scale

from src.extract_data.get_data_from_csv import GetDataFromCSV
from src.pre_processing.extract_landscape import get_facial_vectors
from src.models.facial_feature_based.common import get_normalized_vectors
from src.models.facial_feature_based.common import clean_normalized_vectors


def svc_runner(scaled_clean_normalized_vectors_train,
               clean_targets_train,
               scaled_clean_normalized_vectors_test,
               clean_targets_test,
               gamma_c_pair):
    # ok, we're basically ready to go, split it in to the correct splits
    # and we can train/test.

    classifier = SVC(C=gamma_c_pair[0],
                     gamma=gamma_c_pair[1],
                     probability=True,
                     verbose=True,)
    classifier.fit(scaled_clean_normalized_vectors_train,
                   clean_targets_train)

    train_score = classifier.score(scaled_clean_normalized_vectors_train,
                                   clean_targets_train)
    test_score = classifier.score(scaled_clean_normalized_vectors_test,
                                  clean_targets_test)
    test_predictions = classifier.predict_proba(
        scaled_clean_normalized_vectors_test
    )

    filename = "results/svm_with_gamma_%s_C_%s.csv" % (gamma_c_pair[1], gamma_c_pair[0])
    with open(filename, 'w') as outfile:
        outfile.write("train_score:%s, test_score:%s\n" % (train_score,
                                                           test_score))
        for prediction in test_predictions.tolist():
            outfile.write("%s\n" % str(prediction))
        outfile.flush()


def run():
    """
    Prepares data for and executes a GridSearch-ish search using SVMs, results
    are decent but only when you realise that 1/3 of the data isn't here.
    (dlib does not detect faces for 1/3 of the data)
    :return: Nothing, it'll just take a lot of your CPU power for a while
    and write results to the results folder in a series of CSVs
    """
    print("reading csv data, cached or not")
    csv_reader = GetDataFromCSV()
    facial_pixels_train, targets_train = csv_reader.get_training_data()
    facial_vectors_train = get_facial_vectors(only_train_data=True,
                                              load_cached=True)
    facial_pixels_test, targets_test = csv_reader.get_test_data()
    facial_vectors_test = get_facial_vectors(only_test_data=True,
                                             load_cached=True)

    # get our pixels in to a small vector based on facial features extracted
    # by dlib, gets them all concatenated in a single vector of pixels, after
    # this point, we can discard all other data except for targets.

    print("getting normalized/concatenated facial vectors")
    normalized_vectors_train, feature_target_sizes = get_normalized_vectors(
        facial_vectors_train,
        facial_pixels_train
    )
    normalized_vectors_test, _ = get_normalized_vectors(
        facial_vectors_test,
        facial_pixels_test,
        feature_target_sizes=feature_target_sizes
    )

    # clean a little first
    print("cleaning normalized/concatenated facial vectors up")

    clean_normalized_vectors_train, clean_targets_train, _ = \
        clean_normalized_vectors(
            normalized_vectors_train, targets_train
        )
    clean_normalized_vectors_test, clean_targets_test, bad_index_mask_test = \
        clean_normalized_vectors(
            normalized_vectors_test, targets_test
        )

    scaled_clean_normalized_vectors_train = scale(clean_normalized_vectors_train)
    scaled_clean_normalized_vectors_test = scale(clean_normalized_vectors_test)

    # These are ranges most likely to contain quality values.
    gamma_range = np.logspace(-3, -1, 10)
    C_range = np.logspace(-1, 5, 7)

    arg_pairs = product(C_range, gamma_range)

    pool = multiprocessing.Pool()
    svc_runner_partial = partial(
        svc_runner,
        scaled_clean_normalized_vectors_train,
        clean_targets_train,
        scaled_clean_normalized_vectors_test,
        clean_targets_test,
    )
    print("running")
    pool.map(svc_runner_partial, arg_pairs)
