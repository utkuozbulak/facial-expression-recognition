import dlib
import numpy as np
import os
import logging
from src.extract_data.get_data_from_csv import GetDataFromCSV

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PREDICTOR_PATH = DIR_PATH + "/shape_predictor_68_face_landmarks.dat"
DATA_CSV_FILE = DIR_PATH + '/../../data/fer2013.csv'
TEST_DATA_CACHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_test_data.npy"
TRAIN_DATA_CACHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_train_data.npy"
ALL_DATA_CACHED_PATH = DIR_PATH + "/saved_landscape_processing/landscape_all_data.npy"

IMAGE_SIZE = 48 * 48

TRAIN_END_POINT = 28708
PUBLIC_TEST_START_POINT = 28709
PUBLIC_TEST_END_POINT = 35887
PRIVATE_TEST_END_POINT = 35887

RIGHT_BROW = list(range(17, 22))
LEFT_BROW = list(range(22, 27))
NOSE = list(range(27, 35))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 61))

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def show_example():
    """
    Prints vectors found for first photo in file and shows picture with vectors painted on photo
    """
    photos_matrix = _extract_photos_from_file(DATA_CSV_FILE, True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    img = photos_matrix[0, :, :]
    win = _initialise_gui_window(img)
    dets = detector(img, 1)
    vec = _get_vectors_of_image_from_image(dets, predictor, img, win)
    logger.info(vec)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


def get_facial_vectors(specific_part=None, file_path=DATA_CSV_FILE, only_train_data=False,
                       only_test_data=False, load_cached=False):
    """
    TODO: extract_only_specific_parts

    A list of numpy arrays containing position of specific positions of a face. if no face found a [62,2]
    zero vector is placed in matrix

    Keyword Arguments:
        specific_part -- can specify to return vectors for specified facial features:
                         [RIGHT_BROW, LEFT_BROW, NOSE, RIGHT_EYE, LEFT_EYE, MOUTH]
        only_test_data -- returns facial vectors for only training data
        only_test_data -- returns facial vectors for only test data
        load_cached -- returns image matrix from file if stored, saves processing images again

    :return numpy array [n, 62, 2], where n is number of photos analysed
    """

    def _create_correct_size_facial_matrix(only_train_data, only_test_data):
        if only_train_data:
            facial_featutes_matrix = np.zeros([TRAIN_END_POINT, 68, 2])
        elif only_test_data:
            facial_featutes_matrix = np.zeros([PUBLIC_TEST_END_POINT - TRAIN_END_POINT, 68, 2])
        else:
            facial_featutes_matrix = np.zeros([PRIVATE_TEST_END_POINT, 68, 2])
        return facial_featutes_matrix

    def _save_processed_image_matrix(image_matrix, only_train_data, only_test_data):
        if only_test_data:
            file_name = TEST_DATA_CACHED_PATH
        elif only_train_data:
            file_name = TRAIN_DATA_CACHED_PATH
        else:
            file_name = ALL_DATA_CACHED_PATH
        logger.debug("Saving image_matrix to file {}".format(file_name))
        np.save(file_name, image_matrix)

    def _load_cached_image_matrix(only_train_data, only_test_data):
        if only_test_data:
            file_name = TEST_DATA_CACHED_PATH
        elif only_train_data:
            file_name = TRAIN_DATA_CACHED_PATH
        else:
            file_name = ALL_DATA_CACHED_PATH
        logger.debug("Attempting to load cached file: {}".format(file_name))
        return np.load(file_name)

    if load_cached:
        try:
            image_matrix = _load_cached_image_matrix(only_train_data, only_test_data)
            return image_matrix
        except FileNotFoundError:
            logger.debug("No file found, Processing data")

    logger.debug("Getting facial vectors")
    photos_matrix = _extract_photos_from_file(file_path, False, only_train_data, only_test_data)
    facial_featutes_matrix = _create_correct_size_facial_matrix(only_train_data, only_test_data)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    faces_found = 0
    faces_not_found = 0
    logger.debug("Starting facial detection and feature extraction")
    for i in range(len(photos_matrix)):
        img = photos_matrix[i, :, :]
        dets = detector(img, 2)
        if len(dets) != 0:
            facial_featutes_matrix[i, :, :] = _get_vectors_of_image_from_image(dets, predictor, img)
            faces_found += 1
        else:
            facial_featutes_matrix[i, :, :] = np.zeros([68, 2])
            faces_not_found += 1
        if i % 5000 == 0:
            logger.debug("Photos gone through {}".format(i))

    logger.debug("Faces found: {}  Faces not found : {}".format(faces_found, faces_not_found))
    _save_processed_image_matrix(facial_featutes_matrix, only_train_data, only_test_data)
    return facial_featutes_matrix


def _extract_photos_from_file(file_path=DATA_CSV_FILE, extract_first_only=False, only_train_data=False,
                              only_test_data=False):
    data_getter = GetDataFromCSV()
    if only_train_data:
        images_matrix, _ = data_getter.get_training_data()
    elif only_test_data:
        images_matrix, _ = data_getter.get_test_data()
    elif extract_first_only:
        images_matrix, _ = data_getter.get_first_record()
    else:
        images_matrix, _ = data_getter.get_all_data()
    return images_matrix


def _initialise_gui_window(img):
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    return win


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
    show_example()
