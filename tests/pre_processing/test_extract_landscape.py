import unittest
from pre_processing.extract_landscape import get_facial_vectors
import logging

class TestExtractLandScape(unittest.TestCase):

    logging.basicConfig(level=logging.INFO)
    def test_get_all_vectors(self):
        all_vectors = get_facial_vectors()
        self.assertEqual(all_vectors.shape, (35886,68,2))


if __name__ == '__main__':
    unittest.main()