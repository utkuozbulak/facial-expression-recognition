import dlib
import numpy as np
import os
import logging
from src.format_data import get_data_in_matrix_format
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class FeatureExtract(object):
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    PREDICTOR_PATH = DIR_PATH + "/shape_predictor_68_face_landmarks.dat"
    DATA_CSV_FILE = DIR_PATH + '/../../data/fer2013.csv'
    TEST_DATA_PRIVATE_CACHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_private_test_data.npy"
    TEST_DATA_PUBLIC_CATHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_public_test_data.npy"
    TRAIN_DATA_CACHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_train_data.npy"
    ALL_DATA_CACHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_all_data.npy"

    IMAGE_SIZE = 48 * 48

    RIGHT_BROW = list(range(17, 22))
    LEFT_BROW = list(range(22, 27))
    NOSE = list(range(27, 35))
    RIGHT_EYE = list(range(36, 42))
    LEFT_EYE = list(range(42, 48))
    MOUTH = list(range(48, 61))

    def __init__(self):
        raw_data = pd.read_csv(self.DATA_CSV_FILE)
        emotion = raw_data[['emotion']]
        pixels = raw_data[['pixels']]
        self.x_train_matrix, \
            self.x_public_test_matrix, \
            self.x_private_test_matrix, \
            _, _, _ = get_data_in_matrix_format(emotion, pixels)

    def show_example(self):
        """
        Prints vectors found for first photo in file and shows picture with vectors painted on photo
        """
        img = self.x_train_matrix[1]
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        win = self._initialise_gui_window(img)
        dets = detector(img, 1)
        vec = self._get_vectors_of_image_from_image(dets, predictor, img, win)
        logger.info(vec)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()

    def get_test_data(self, load_cached=False):
        """
        A list of numpy arrays containing position of specific positions of a face. if no face found a [62,2]
        zero vector is placed in matrix

        :param load_cached: returns image matrix from file if stored, saves processing images again
        :return 2 numpy arrays [public, private] in format [n, 62, 2], where n is number of photos analysed,
        """

        if load_cached:
            try:
                facial_features_test_public = self._load_cached_image_matrix(self.TEST_DATA_PUBLIC_CATHED_PATH)
                facial_features_test_private = self._load_cached_image_matrix(self.TEST_DATA_PRIVATE_CACHED_PATH)
                return facial_features_test_public, facial_features_test_private
            except FileNotFoundError:
                logger.debug("No file found, Processing data")

        facial_features_test_public = self._get_facial_vectors(self.x_public_test_matrix)
        facial_features_test_private = self._get_facial_vectors(self.x_private_test_matrix)
        self._save_processed_image_matrix(self.TEST_DATA_PUBLIC_CATHED_PATH, facial_features_test_public)
        self._save_processed_image_matrix(self.TEST_DATA_PRIVATE_CACHED_PATH, facial_features_test_private)
        return facial_features_test_public, facial_features_test_private

    def get_training_data(self, load_cached=False):
        """
        A list of numpy arrays containing position of specific positions of a face. if no face found a [62,2]
        zero vector is placed in matrix

        :param load_cached: returns image matrix from file if stored, saves processing images again
        :return numpy array [n, 62, 2], where n is number of photos analysed
        """

        if load_cached:
            try:
                facial_features_training = self._load_cached_image_matrix(self.TRAIN_DATA_CACHED_PATH)
                return facial_features_training
            except FileNotFoundError:
                logger.debug("No file found, Processing data")

        facial_features_train = self.x_train_matrix
        facial_features = self._get_facial_vectors(facial_features_train)
        self._save_processed_image_matrix(self.TRAIN_DATA_CACHED_PATH, facial_features)
        return facial_features

    def _get_facial_vectors(self, photos_matrix):

        logger.debug("Getting facial vectors")
        facial_featutes_matrix = np.zeros([len(photos_matrix), 68, 2])
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        faces_found, faces_not_found = 0, 0
        logger.debug("Starting facial detection and feature extraction")
        for i in range(len(photos_matrix)):
            img = photos_matrix[i, :, :]
            dets = detector(img, 2)
            if len(dets) != 0:
                facial_featutes_matrix[i, :, :] = self._get_vectors_of_image_from_image(dets, predictor, img)
                faces_found += 1
            else:
                facial_featutes_matrix[i, :, :] = np.zeros([68, 2])
                faces_not_found += 1
            if i % 5000 == 0:
                logger.debug("Photos gone through {}".format(i))

        logger.debug("Faces found: {}  Faces not found : {}".format(faces_found, faces_not_found))
        return facial_featutes_matrix

    @staticmethod
    def _save_processed_image_matrix(file_name, image_matrix):
        np.save(file_name, image_matrix)

    @staticmethod
    def _load_cached_image_matrix(file):
        logger.debug("Attempting to load cached file: {}".format(file))
        return np.load(file)

    @staticmethod
    def _initialise_gui_window(img):
        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(img)
        return win

    @staticmethod
    def _get_vectors_of_image_from_image(dets, predictor, img, win=None):
        for k, d in enumerate(dets):
            shape = predictor(img, d)
        vec = np.empty([68, 2], dtype=int)
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
        if win:
            win.add_overlay(shape)
        return vec


if __name__ == "__main__":
    fe = FeatureExtract()
    fe.get_test_data()
    fe.get_training_data(load_cached=True)
