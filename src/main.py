import pandas as pd

from format_data import get_data_in_vectoral_format, csv2array, get_data_in_matrix_format
from data2img import export_image
from svd import save_sample_decomposition, decompose, generate_decomposed_matrices, generate_decomposed_matrices_from_list


raw_data_csv_file_name = '../data/fer2013.csv'

if __name__ == "__main__":
    raw_data = pd.read_csv(raw_data_csv_file_name)
    emotion = raw_data[['emotion']]
    pixels = raw_data[['pixels']]
    
    # Data formatting, vectoral and matrix form
    # Vectoral
    ( x_train_vectoral, x_public_test_vectoral, x_private_test_vectoral,
    y_train, y_public_test, y_private_test) = get_data_in_vectoral_format(emotion, pixels)
    # Matrix
    (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
     y_train, y_public_test, y_private_test) = get_data_in_matrix_format(emotion, pixels)

    # Image export
    Img_data = csv2array(emotion, pixels)
    export_image(Img_data)

    # Machine Learning

    # SVD - Decomposed images
    # BEWARE ! The line below takes incredibly long to run because it is calculating a decomposition
    # For each image, hence its commented out for now
    # decomposed_image_list = generate_decomposed_matrices_from_list(x_train_matrix)
    # An example decomposition
    example_image = x_train_matrix[4]  # Example image from matrix form, e.g: fourth image
    U,S,V = decompose(example_image)  # Decomposition
    decomposed_list = generate_decomposed_matrices(U, S, V)  # Decomposed list, 48 matrices
    # You can play with the number '10' and observe different results
    # As the sum of decompositions increase the summed image gets better and better
    save_sample_decomposition(decomposed_list, 10)  # Saves some sample results of this decomposition
