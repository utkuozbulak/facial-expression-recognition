from src.extract_data.get_data_from_csv import GetDataFromCSV
import unittest


class TestExtractLandScape(unittest.TestCase):
    def test_csv_extract_only_train_images(self):
        data_extractor = GetDataFromCSV()
        extracted_images, _ = data_extractor.get_training_data()
        self.assertEqual(extracted_images.shape, (28708, 48, 48))

    def test_csv_extract_only_test_images(self):
        data_extractor = GetDataFromCSV()
        extracted_images, _ = data_extractor.get_test_data()
        self.assertEqual(extracted_images.shape, (35887 - 28709, 48, 48))
