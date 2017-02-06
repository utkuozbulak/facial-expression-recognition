import csv
import os
import numpy as np
from scipy.misc import toimage

IMG_SIZE = 48
CSV_FILE_NAME = '../data/head.csv'
DATA_OUTPUT_DIR = '../data/img'


# based on https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
# based on https://stackoverflow.com/questions/434583/what-is-the-fastest-way-to-draw-an-image-from-discrete-pixel-values-in-python
def csv2array(file_name, img_size=IMG_SIZE):
    """
    converts rows of the fer2013.csv file into a list of numpy arrays, representing the image

    :param img_size: number of pixels in image, defaults to 48, assumes square image
    :param file_name: csv file name
    :return: list of tuples (expression, numpyarray)
    """
    img_list = []
    csv_file = csv.DictReader(open(file_name))

    for csv_row in csv_file:
        data = np.zeros((img_size, img_size), dtype=np.uint8)
        pixel_data = csv_row['pixels'].split()
        for i in range(0, img_size):
            pixel_index = i * img_size
            data[i] = pixel_data[pixel_index:pixel_index + img_size]
        img_list.append((csv_row['emotion'], data))

    return img_list


def display_img(data):
    i = toimage(data[1])
    i.show()


def save_img(data, file_name):
    i = toimage(data[1])
    i.save(file_name)


if __name__ == "__main__":
    # convert image to data
    # returns data in format [(expression, numpy array)]
    img_data = csv2array(CSV_FILE_NAME)

    if not os.path.exists(DATA_OUTPUT_DIR):
        os.makedirs(DATA_OUTPUT_DIR)
    # save first 10 images for demo purpose
    for k in range(0, 9):
        img = img_data[k]
        # save image with filename "index_emotion.png"
        save_img(img, DATA_OUTPUT_DIR + "/{}_{}.png".format(k, img[0]))

    # display an example for demo purpose
    display_img(img_data[2])
