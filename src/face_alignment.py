import os
import pandas as pd
from scipy.misc import imread
from data2img import export_training_and_test_images


def align_faces(img_data):
    export_training_and_test_images(img_data)

    # Run docker script (???) to perform the face alignment
    # and export the aligned images to a new dir

    # This seems to be somewhat complicated, but for now one can follow
    # the instructions on issue #12 for how to create a dir with the aligned faces.

    # There will be csv file (train_aligned.csv) with the end result (aligned faces) available for download.


def images_to_array(img_dir):
    """
    Takes a directory with images and transform them into pixel arrays
    :param: img_dir is directory of images to transform into pixel values
    :return: Array of tuples with emotion and numpy.ndarray flattened to one dimensions
    """
    emotion_and_pixels = []

    for emotion in range(7):
        for root, dir_names, file_names in os.walk(img_dir + str(emotion)):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                pixel_array = imread(file_path)  # 48*48 array
                pixels_flat = " ".join([str(pixel) for pixel in pixel_array.flatten()])
                emotion_and_pixels.append((emotion, pixels_flat))

    return emotion_and_pixels


def array_to_csv(images_array, output_file):
    emotion_and_pixels_df = pd.DataFrame(images_array, columns=['emotion', 'pixels'])
    emotion_and_pixels_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    imgs_array = images_to_array('../data/img/train_aligned/')
    array_to_csv(imgs_array, '../data/train_aligned.csv')
