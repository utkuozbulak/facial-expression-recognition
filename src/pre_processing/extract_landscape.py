import dlib
import numpy as np
import os
import csv
import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
PREDICTOR_PATH = dir_path + "/shape_predictor_68_face_landmarks.dat"
DATA_CSV_FILE = dir_path + '/../data/fer2013.csv'
IMAGE_SIZE = 48 * 48

TRAIN_END_POINT = 28708
PUBLIC_TEST_START_POINT = 28709
PUBLIC_TEST_END_POINT = 32297
PRIVATE_TEST_END_POINT = 32297

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
    photos_matrix = _extract_photos_from_file(True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    img = photos_matrix[0, :, :]
    win = _initilise_gui_window(img)
    dets = detector(img, 1)
    vec = _get_vectors_of_image_from_image(dets, predictor, img, win)
    logger.info(vec)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


def get_facial_vectors(specific_part=None, file_path=PREDICTOR_PATH, only_train_data=False, only_test_data=False):
    """
    TODO: Make it work with extracting only test and train data
    TODO: get precompiled data if availible

    A list of numpy arrays containing position of specific positions of a face. if no face found a [62,2]
    zero vector is placed in matrix

    Keyword Arguments:
        specific_part -- can specify to return vectors for specified facial features:
                         [RIGHT_BROW, LEFT_BROW, NOSE, RIGHT_EYE, LEFT_EYE, MOUTH]
        only_test_data -- returns facial vectors for only training data
        only_test_data -- returns facial vectors for only test data

    :return numpy array [n, 62, 2], where n is number of photos analysed
    """
    logger.debug("Getting facial vectors")
    photos_matrix = _extract_photos_from_file(False, only_train_data, only_test_data)

    facial_featutes_matrix = np.zeros([35886, 68, 2])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    faces_found = 0
    faces_not_found = 0
    logger.debug("Starting facial detection and feature extraction")
    for i in range(35886):
        img = photos_matrix[i, :, :]
        dets = detector(img, 2)
        if len(dets) != 0:
            facial_featutes_matrix[i, :, :] = _get_vectors_of_image_from_image(dets, predictor, img)
            faces_found += 1
        else:
            facial_featutes_matrix[i, :, :] = np.zeros([68, 2])
            faces_not_found += 1
        if i % 10000 == 0:
            logger.debug("photos gone through {}".format(i))

    print("faces found: {}   faces not found : {}".format(faces_found, faces_not_found))

    return facial_featutes_matrix


def _extract_photos_from_file(extract_first_only=False, only_train_data=False, only_test_data=False):
    logger.debug("Extracting photos from csv")
    if only_train_data:
        images_matrix = np.zeros([TRAIN_END_POINT, 48, 48], dtype="uint8")
    elif only_test_data:
        images_matrix = np.zeros([PUBLIC_TEST_END_POINT - 28708, 48, 48], dtype="uint8")
    else:
        images_matrix = np.zeros([35886, 48, 48], dtype="uint8")

    with open(DATA_CSV_FILE, newline='') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for k, pixels in enumerate(reader):
            k_mod = k
            image_csv_format = np.zeros([1, IMAGE_SIZE])
            pixels_formated = [int(a) for a in pixels[1].split(" ")]
            image_csv_format[0, :] = pixels_formated
            image_csv_format = np.reshape(image_csv_format, [48, 48])
            if k % 5000 == 0:
                logger.debug("photos extracted from csv {}".format(k))
            if only_train_data and k > TRAIN_END_POINT:
                break
            elif only_test_data & k < PUBLIC_TEST_START_POINT:
                next
                k_mod = k - PUBLIC_TEST_START_POINT
            images_matrix[k_mod, :, :] = image_csv_format
            if extract_first_only:
                break

    logger.debug("Finished extracting photos from csv")
    return images_matrix


def _initilise_gui_window(img):
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
