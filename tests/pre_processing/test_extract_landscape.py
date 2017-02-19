import logging
import unittest
import os.path

from src.pre_processing.extract_landscape import FeatureExtract

class TestExtractLandScape(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)

    def setUp(self):
        self.fe = FeatureExtract()

    def test_get_training_data_vectors(self):
        training_vectors = self.fe.get_training_data(load_cached=True)
        self.assertEqual(training_vectors.shape, (28708, 68, 2))

    def test_get_test_data_only(self):
        test_vectors_public, test_vectors_private = self.fe.get_test_data(load_cached=True)
        self.assertEqual(test_vectors_public.shape, (3588, 68, 2))
        self.assertEqual(test_vectors_private.shape, (3590, 68, 2))


if __name__ == '__main__':
    unittest.main()
