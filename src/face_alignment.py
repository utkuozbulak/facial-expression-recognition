import os
from scipy.misc import imread
from data2img import export_training_and_test_images


def align_faces(img_data):
    export_training_and_test_images(img_data)

    # Run docker script (???) to perform the face alignment
    # and export the aligned images to a new dir
    return None


def images_to_array(img_dir):
    """
    Takes a directory with images and transform them into pixel arrays
    :param: img_dir is directory of images to transform into pixel values
    :return: Array with numpy.ndarrays flattened to one dimensions
    """
    pixels = []
    for root, dir_names, file_names in os.walk(img_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            pixel_array = imread(file_path)  # 48*48 array
            pixels_flat = pixel_array.flatten()  # Flatten 2D array to 1D array
            pixels.append(pixels_flat)
    return pixels
