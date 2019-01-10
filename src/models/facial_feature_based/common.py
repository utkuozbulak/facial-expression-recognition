import numpy as np

from src.pre_processing.image_vectorization import get_concatenated_vectors
from src.pre_processing.image_vectorization import get_resized_feature_vectors


def get_normalized_vectors(facial_vectors,
                           facial_pixels,
                           feature_target_sizes=None):
    """
    convenience wrapper around `get_resized_feature_vectors` and
    `get_concatenated_vectors`, see their docs for specifics.
    :param facial_vectors: Vectors of the facial features, as provided by dlib
    :param facial_pixels: the pixels from the csv.
    :param feature_target_sizes: there are times we need to tell the
    normalization algo what feature sizes to scale to (i.e: when using train
    data to normalize test data to the same scale because we shouldn't rely on
    test data in advance.
    :return: the normalized/concatenated vectors, see `get_concatenated_vectors`
    for more details on that one, also returns the normalization vector used,
    which will simply be the feature_target_sizes parameter if provided, or the
    one calculated/generated from your images if not (i.e: to re-feed for the
    test data)
    """

    # facial_vectors.view(np.uint8)

    normalized_facial_features, target_feature_sizes = get_resized_feature_vectors(
        facial_vectors,
        facial_pixels,
        feature_target_sizes=feature_target_sizes
    )
    normalized_concatenated_facial_features_matrix = get_concatenated_vectors(
        normalized_facial_features,
    )

    return normalized_concatenated_facial_features_matrix, target_feature_sizes


def clean_normalized_vectors(normalized_vectors, targets):
    """
    The purpose of this method is to synchronize un-normalizable images (when
    no face is detected) and the targets, so we can feed the recognized faces
    in to whatever we want.
    :param normalized_vectors: the normalized vectors, with Nones for
     unnormalizable (default of `get_normalized_vectors` should do this.
    :param targets: the targets matrix, as read from the CSV
    :return: normalized_vectors with None's stripped out and the targets array
    with the corresponding targets stripped out (we're aligning things),
    finally, the good/bad index mask, for convenience (to
    e.g: retrieve those indices and feed them separately to a fallback algo)
    """
    bad_index_mask = np.equal(normalized_vectors, None)
    good_index_mask = np.not_equal(normalized_vectors, None)

    good_normalized_elements = normalized_vectors[good_index_mask]
    good_targets = targets[good_index_mask]

    good_normalized_elements = np.vstack(good_normalized_elements)

    return good_normalized_elements, good_targets, bad_index_mask
