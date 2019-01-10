import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from src.extract_data.get_data_from_csv import GetDataFromCSV
from src.pre_processing.extract_landscape import get_facial_vectors

from src.models.facial_feature_based.common import get_normalized_vectors
from src.models.facial_feature_based.common import clean_normalized_vectors


def run():
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
    clean_normalized_vectors_train, clean_targets_train, _ = clean_normalized_vectors(
        normalized_vectors_train, targets_train
    )
    clean_normalized_vectors_test, clean_targets_test, _ = clean_normalized_vectors(
        normalized_vectors_test, targets_test
    )

    scaled_clean_normalized_vectors_train = scale(clean_normalized_vectors_train)
    scaled_clean_normalized_vectors_test = scale(clean_normalized_vectors_test)

    n_neighbors_log = np.floor(np.logspace(1, 3, 20))
    n_neighbors_log = n_neighbors_log[n_neighbors_log>21]
    n_neighbors_log = n_neighbors_log.astype(int)

    for n_neighbors in n_neighbors_log:
        # ok, we're basically ready to go, split it in to the correct splits
        # and we can train/test.
        classifier = KNeighborsClassifier(weights='distance',
                                          n_jobs=-2,
                                          n_neighbors=n_neighbors)
        classifier.fit(scaled_clean_normalized_vectors_train,
                       clean_targets_train)

        train_score = classifier.score(scaled_clean_normalized_vectors_train,
                                       clean_targets_train)
        test_score = classifier.score(scaled_clean_normalized_vectors_test,
                                      clean_targets_test)

        test_predictions = classifier.predict(scaled_clean_normalized_vectors_test)

        filename = "results/knn_%s.csv" % n_neighbors
        with open(filename, 'w') as outfile:
            outfile.write("train_score:%s, test_score:%s\n" % (train_score,
                                                               test_score))
            for prediction in test_predictions.tolist():
                outfile.write("%s\n" % str(prediction))
            outfile.flush()
