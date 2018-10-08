from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
from format_data import get_data_in_matrix_format
from data2img import save_single_img, get_zoomed_image

raw_data_csv_file_name = '../../../data/fer2013.csv'


def get_nnmf_features(pictures=None):

    if pictures is None:
        raw_data = pd.read_csv(raw_data_csv_file_name)
        emotion = raw_data[['emotion']]
        pixels = raw_data[['pixels']]
        # Get data in matrix form
        (x_train_matrix, x_public_test_matrix, x_private_test_matrix,
         y_train, y_public_test, y_private_test) = get_data_in_matrix_format(emotion, pixels)
        # Only used train and public test for now
        pictures = x_train_matrix.astype('float32')

    return nnmf_decomposition(pictures)


def save_sample_decomposition(w,h, first_n_decompositions):
    """
    # Saves a sample decomposition
    """
    image = w[:, 0:first_n_decompositions].dot(h[0:first_n_decompositions, :])
    summed_image = get_zoomed_image(image, 500)  # 500 percentage zoom
    save_single_img(summed_image, 'summed_image')


def nnmf_decomposition(pictures):
    model = NMF(n_components=6, init='random', random_state=0)
    nnmf_facial_features = np.zeros((len(pictures), 48, 6))
    for idx, pic in enumerate(pictures):
        w = model.fit_transform(pic)
        h = model.components_
        nnmf_facial_features[idx, :] = w
    return nnmf_facial_features


if __name__ == "__main__":
    get_nnmf_features()


