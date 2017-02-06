import numpy as np
import scipy.misc
import csv
import os

IMG_SIZE = 48
csv_file_name = '../data/head.csv'
data_output_dir = '../data/img'


# based on https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
# based on https://stackoverflow.com/questions/434583/what-is-the-fastest-way-to-draw-an-image-from-discrete-pixel-values-in-python
def csv2array(file_name):
    img_list = []
    csv_file = csv.DictReader(open(file_name))

    for csv_row in csv_file:
        data = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        pixel_data = csv_row['pixels'].split()
        for i in range(0, IMG_SIZE):
            pixel_index = i * IMG_SIZE
            data[i] = pixel_data[pixel_index:pixel_index + IMG_SIZE]
        img_list.append((csv_row['emotion'], data))

    return img_list


def display_img(data):
    i = scipy.misc.toimage(data[1])
    i.show()


def save_img(data, file_name):
    i = scipy.misc.toimage(data[1])
    i.save(file_name)


if __name__ == "__main__":
    # convert image to data
    # returns data in format [(expression, numpy array)]
    img_data = csv2array(csv_file_name)

    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)
    # save first 10 images for demo purpose
    for k in range(0, 9):
        img = img_data[k]
        # save image with filename "index_emotion.png"
        save_img(img, data_output_dir + "/{}_{}.png".format(k, img[0]))

    # display an example for demo purpose
    display_img(img_data[2])
