import os
from scipy.misc import toimage, imresize

from format_data import TRAIN_END


DATA_OUTPUT_DIR = '../data/img'
DECOMPOSITION_OUTPUT_DIR = '../data/img/decomposition/'
DATA_TRAINING_OUTPUT_DIR = '../data/img/train'
DATA_TEST_OUTPUT_DIR = '../data/img/test'


def display_img(data):
    i = toimage(data[1])
    i.show()


def save_img(data, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    i = toimage(data[1])
    i.save(file_name)


def export_image(img_data):
    for i, img in enumerate(img_data):
        # save image with filename "/{expression}/{row_num}.png"
        save_img(img, DATA_OUTPUT_DIR + "/{}/{}.png".format(img[0], i))
    # display an example for demo purpose
    display_img(img_data[2])


def export_training_and_test_images(img_data):
    for i, img in enumerate(img_data):
        if i <= TRAIN_END:
            save_img(img, DATA_TRAINING_OUTPUT_DIR + "/{}/{}.png".format(img[0], i))
        else:
            save_img(img, DATA_TEST_OUTPUT_DIR + "/{}/{}.png".format(img[0], i))


def show_zoomed_image(image_as_list ,zoom_ratio):
    resized_list = imresize(image_as_list, zoom_ratio)
    resized_img = toimage(resized_list)
    resized_img.show()
    return resized_img


def get_zoomed_image(image_as_list ,zoom_ratio):
    resized_list = imresize(image_as_list, zoom_ratio)
    resized_img = toimage(resized_list)
    return resized_img


def save_single_img(img, file_name):
    if not os.path.exists(os.path.dirname(DECOMPOSITION_OUTPUT_DIR)):
        os.makedirs(os.path.dirname(DECOMPOSITION_OUTPUT_DIR))
    img.save(DECOMPOSITION_OUTPUT_DIR+ '/'+file_name+'.png')
