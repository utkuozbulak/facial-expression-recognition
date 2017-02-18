from keras.utils import np_utils
import numpy as np


TRAIN_END = 28708
PUBLIC_TEST_START = 28709
PUBLIC_TEST_END = 32297
PRIVATE_TEST_START = 32297
IMG_SIZE = 48


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def get_data_in_vectoral_form(emotion, pixels):
    emotion_as_list = pandas_vector_to_list(emotion)
    pixels_as_list = pandas_vector_to_list(pixels)
    x_vectoral_data = []
    y_data = []
    for index,item in enumerate(pixels_as_list):
        single_vector = []
        pixels = item
        single_vector = [int(data) for data in pixels.split()]
        x_vectoral_data.append(single_vector)
        y_data.append(emotion_as_list[index])
    # X data
    x_train_vectoral = np.array(x_vectoral_data[0:TRAIN_END])
    x_public_test_vectoral  = np.array(x_vectoral_data[PUBLIC_TEST_START:PUBLIC_TEST_END])
    x_private_test_vectoral = np.array(x_vectoral_data[PRIVATE_TEST_START:])
    y_data_categorical = np_utils.to_categorical(y_data, 7)
    # Y data
    y_train = np.array(y_data_categorical[0:TRAIN_END])
    y_public_test = np.array(y_data_categorical[PUBLIC_TEST_START:PUBLIC_TEST_END])
    y_private_test = np.array(y_data_categorical[PRIVATE_TEST_START:])
    return (x_train_vectoral, x_public_test_vectoral, x_private_test_vectoral,
            y_train, y_public_test, y_private_test)  # Returns data in vectoral format


def csv2array(emotion, pixels, img_size=IMG_SIZE):
    emotion_as_list = pandas_vector_to_list(emotion)
    pixels_as_list = pandas_vector_to_list(pixels)
    img_list = []
    for index,item in enumerate(pixels_as_list):
        data = np.zeros((img_size, img_size), dtype=np.uint8)
        pixel_data = item.split()
        for i in range(0, img_size):
            pixel_index = i * img_size
            data[i] = pixel_data[pixel_index:pixel_index + img_size]
        img_list.append((emotion_as_list[index], data))
    return img_list  # Returns a tuple, emotion:pixel array


def get_data_in_matrix_format(emotion, pixels):
    img_list = csv2array(emotion, pixels)
    image_array = []
    y_data = []
    for item in img_list:
        y_data.append(item[0])
        image_array.append(np.array(item[1]))
    np_image_array = np.array(image_array)
    # X data
    x_train_matrix = np.array(np_image_array[0:TRAIN_END])
    x_public_test_matrix  = np.array(np_image_array[PUBLIC_TEST_START:PUBLIC_TEST_END])
    x_private_test_matrix = np.array(np_image_array[PRIVATE_TEST_START:])
    # Y data
    y_data_categorical = np_utils.to_categorical(y_data, 7)
    y_train = np.array(y_data_categorical[0:TRAIN_END])
    y_public_test = np.array(y_data_categorical[PUBLIC_TEST_START:PUBLIC_TEST_END])
    y_private_test = np.array(y_data_categorical[PRIVATE_TEST_START:])
    
    return (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
            y_train, y_public_test, y_private_test)  # Returns data in matrix format
