import numpy as np
from data2img import save_single_img, get_zoomed_image


def decompose(image_matrix):
    """
    # Get decomposed matrices, USVtranspose
    """
    U, S, V = np.linalg.svd(image_matrix, full_matrices=True)
    reshaped_u = reshape_matrix(U)
    return reshaped_u, S, V


def reshape_matrix(matrix):
    """
    # Transpose of sorts, was needed because of the way arrays are stored
    # Different than MATLAB
    """
    reshaped_matrix = []
    for i in range(0, len(matrix[0])):
        vector = matrix[:,i]
        reshaped_matrix.append(vector)
    reshaped_matrix = np.array(reshaped_matrix)
    return reshaped_matrix


def vector_vector_transpose_multiplication(vector1, vector2):
    """
    # Generates matrix
    """
    result_matrix = []
    for item in vector1:
        single_vector = item * vector2
        single_vector = np.array(single_vector)
        result_matrix.append(single_vector)
    result_matrix = np.array(result_matrix)
    return result_matrix


def decomposed_matrix_multiplication(vector1, scalar, vector2):
    """
    # Generates a single matrix from vector vector transpose and scalar multiplication
    """
    matrix = vector_vector_transpose_multiplication(vector1, vector2)
    matrix = scalar * matrix
    return matrix


def generate_decomposed_matrices_from_list(image_list):
    """
    # Generates a list of decompositions
    # BEWARE ! If the list is long, this take time
    """
    decomposed_list = []
    for item in image_list:
        U,S,V = decompose(item)
        decomposed_matrices = generate_decomposed_matrices(U,S,V)
        decomposed_list.append(decomposed_matrices)
    return decomposed_list


def save_sample_decomposition(decomposed_list, first_n_decompositions):
    """
    # Saves a sample decomposition
    """
    summed_image = np.zeros((48,48))
    for index,item in enumerate(decomposed_list[:first_n_decompositions:]):  # First n decompositions
        image = get_zoomed_image(item, 500)  # 500 percentage zoom
        save_single_img(image, str(index))
        summed_image = summed_image + item
    summed_image = get_zoomed_image(summed_image, 500)  # 500 percentage zoom
    save_single_img(summed_image, 'summed_image')


def generate_decomposed_matrices(U,S,V):
    """
    # Generates all sub decomposition matrices from a SVD
    """
    decomposed_matrices = []
    for i in range(0,len(S)):
        single_matrix = decomposed_matrix_multiplication(U[i], S[i], V[i])
        decomposed_matrices.append(single_matrix)
    return decomposed_matrices
