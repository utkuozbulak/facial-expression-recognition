import logging
import unittest
import os.path

from src.pre_processing.extract_landscape import get_facial_vectors, _extract_photos_from_file

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_CACHED_PATH = DIR_PATH + "/../../src/pre_processing/saved_landscape_processing/landscape_test_data.npy"
TRAIN_DATA_CACHED_PATH = DIR_PATH + "/../../src/pre_processing/saved_landscape_processing/landscape_train_data.npy"
ALL_DATA_CACHED_PATH = DIR_PATH + "/../../src/pre_processing/saved_landscape_processing/landscape_all_data.npy"


class TestExtractLandScape(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)

    def test_csv_extract_only_train_images(self):
        extracted_images = _extract_photos_from_file(only_train_data=True)
        self.assertEqual(extracted_images.shape, (28708, 48, 48))

    def test_csv_extract_only_test_images(self):
        extracted_images = _extract_photos_from_file(only_test_data=True)
        self.assertEqual(extracted_images.shape, (35887 - 28708, 48, 48))

    def test_get_training_data_vectors(self):
        training_vectors = get_facial_vectors(only_train_data=True)
        self.assertEqual(training_vectors.shape, (28708, 68, 2))

    def test_get_test_data_only(self):
        test_vectors = get_facial_vectors(only_test_data=True)
        self.assertEqual(test_vectors.shape, (7179, 68, 2))

    def test_get_all_vectors(self):
        all_vectors = get_facial_vectors()
        self.assertEqual(all_vectors.shape, (35887, 68, 2))

    def test_get_cached_training_data(self):
        self.assertTrue(os.path.isfile(TRAIN_DATA_CACHED_PATH), "No file to load data from")
        training_vectors = get_facial_vectors(only_train_data=True, load_cached=True)
        self.assertEqual(training_vectors.shape, (28708, 68, 2))

    def test_get_cached_test_data(self):
        self.assertTrue(os.path.isfile(TEST_DATA_CACHED_PATH), "No file to load data from")
        training_vectors = get_facial_vectors(only_test_data=True, load_cached=True)
        self.assertEqual(training_vectors.shape, (35887 - 28708, 68, 2))

    def test_get_cached_all_data(self):
        self.assertTrue(os.path.isfile(ALL_DATA_CACHED_PATH), "No file to load data from")
        training_vectors = get_facial_vectors(load_cached=True)
        self.assertEqual(training_vectors.shape, (35887, 68, 2))


if __name__ == '__main__':
    unittest.main()
