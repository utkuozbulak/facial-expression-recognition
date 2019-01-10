import numpy as np
from scipy.misc import imresize

from src.pre_processing.extract_landscape import RIGHT_BROW
from src.pre_processing.extract_landscape import LEFT_BROW
from src.pre_processing.extract_landscape import NOSE
from src.pre_processing.extract_landscape import RIGHT_EYE
from src.pre_processing.extract_landscape import LEFT_EYE
from src.pre_processing.extract_landscape import MOUTH


FACIAL_FEATURE_SETS = {
    "right_brow": RIGHT_BROW,
    "left_brow": LEFT_BROW,
    "nose": NOSE,
    "right_eye": RIGHT_EYE,
    "left_eye": LEFT_EYE,
    "mouth": MOUTH,
}


def _get_feature_shape(feature_points, facial_vector):
    """
    get the maximum vertical and horizontal sizes of a particular feature,
    given a set of coordinates will simply calculate:
    max(x-axis) - min(x-axis)
    max(y-axis) - min(x_axis)
    :param feature_points: the particular points of the feature you're
    trying to find the size of, i.e: points 4,5,6 - these are available in
    FACIAL_FEATURE_SETS
    :param facial_vector: the facial vector from which to get the min/max
    of these points.
    :return: the x and y sizes.
    """
    interesting_data_x = facial_vector[feature_points, 0]
    interesting_data_y = facial_vector[feature_points, 1]

    x_size = max(interesting_data_x) - min(interesting_data_x)
    y_size = max(interesting_data_y) - min(interesting_data_y)

    return int(x_size), int(y_size)


def _get_min_max_feature_numbers(feature_points, facial_vector):
    """
    works similarly to `_get_feature_shape` except that this returns the
    points that cause the max dimensions - that's two for each axis.
    :param feature_points: the particular points of the feature you're
    trying to find the size of, i.e: points 4,5,6 - these are available in
    FACIAL_FEATURE_SETS
    :param facial_vector: the facial vector from which to get the min/max
    of these points.
    :return: xmin_feature_number,
             xmax_feature_number,
             ymin_feature_number,
             ymax_feature_number
    note that you can use the feature numbers to obtain coordinates.
    """
    minx = feature_points[facial_vector[feature_points, 0].argmin()]
    maxx = feature_points[facial_vector[feature_points, 0].argmax()]
    miny = feature_points[facial_vector[feature_points, 1].argmin()]
    maxy = feature_points[facial_vector[feature_points, 1].argmax()]

    return minx, maxx, miny, maxy


def get_largest_features(facial_vectors):
    """
    This method looks through facial vectors provided for the largest eyes,
    mouth, nose, etc in order to scale images up to.
    :param facial_vectors: the dlib extracted factial vectors
    :return: the largest x and y size of each feature, in a dict with an 'x'
    and 'y' value for each feature name, i.e: return_value['right_brow']['x']
    """
    max_feature_size = {
        "right_brow": {"x": 0, "y": 0},
        "left_brow": {"x": 0, "y": 0},
        "nose": {"x": 0, "y": 0},
        "right_eye": {"x": 0, "y": 0},
        "left_eye": {"x": 0, "y": 0},
        "mouth": {"x": 0, "y": 0},
    }
    for face in facial_vectors:
        if face.any():
            for feature_key in FACIAL_FEATURE_SETS.keys():
                x_size, y_size = _get_feature_shape(
                    FACIAL_FEATURE_SETS[feature_key], face
                )
                max_feature_size[feature_key]['x'] = max(
                    x_size, max_feature_size[feature_key]['x']
                )
                max_feature_size[feature_key]['y'] = max(
                    y_size, max_feature_size[feature_key]['y']
                )
    return max_feature_size


def _get_resized_feature_vector(feature_data,
                               feature_target_size,
                               pixels,
                               maintain_aspect_ratio=False):
    """
    take in an image and the facial vector and resize each feature to fit in to
    the largest possible space for that feature. This method will change
    through experimentation, right now it's basic resizing.
    :param feature_data: the data provided by dlib facial feature extraction
    should be a 68x2 vector.
    :param feature_target_size: the data straight-up provided by
    `get_largest_features` - this is used to resize all images to the params
    specified by that dict.
    :param pixels: pixels of the image, should be 48x48 or similar.
    :param maintain_aspect_ratio: what it says on the tin. black-padded.
    :return: the resized features in a dict, each resized to the largest
    possible feature, this size is obtained from `get_largest_features`.
    BEWARE: will insert a None in this list instead of a dict if there is
    no face data from dlib. handle your Nones yo.
    """
    resized_image_features = {}
    # clip feature data, this is probably wrong:
    # TODO: improve this.
    clipped_feature_data = feature_data.clip(0, 48)

    for feature_key in FACIAL_FEATURE_SETS.keys():
        # grab the feature out of the full image. how hard could it be?
        (minx_index, maxx_index,
         miny_index, maxy_index) = _get_min_max_feature_numbers(
            FACIAL_FEATURE_SETS[feature_key],
            clipped_feature_data
        )

        # now grab out the x/y's of the pixels - separated for some clarity.
        minx = int(clipped_feature_data[minx_index, 0])
        maxx = int(clipped_feature_data[maxx_index, 0]) + 1
        miny = int(clipped_feature_data[miny_index, 1])
        maxy = int(clipped_feature_data[maxy_index, 1]) + 1

        # ok, we have our data for this facial feature,
        # (scale and index of min/max points on each axis),
        # let's extract that part of the image and get it rescaled.
        # this is simple, use the min/max values
        feature_pixels = pixels[minx:maxx, miny:maxy]

        # kk resize dis up to our feature_target_size.
        target_size = (feature_target_size[feature_key]['x'],
                       feature_target_size[feature_key]['y'])
        resized_feature = imresize(feature_pixels, target_size)
        resized_image_features[feature_key] = resized_feature

    return resized_image_features


def get_resized_feature_vectors(feature_data,
                                pixels,
                                maintain_aspect_ratio=False,
                                feature_target_sizes=None):
    """
    Bulk processor for taking in feature data (the mouth, nose, ears thing)
    and pumping images of each of these parts to the size of the largest
    x and y axes of each feature... strange to explain.
    :param feature_data: the features extracted, should be [num_images]x68x2
    :param pixels: pixels of the image, should be [num_images]x48x48 or similar
    :param maintain_aspect_ratio: currently unused, but will add black padding
    at a later date.
    :return: a list of dicts, where each dict is the pixels of the rescaled
    feature - the same features for any input image _should_ then be the same
    size and feedable to a neural net. This is a simple resize. experimentation
    is due.
    Also returns the sizes we mapped to, to be re-fed back in to this function
    for resizing test data to same size as train data (we can't "know" test
    data in advance, can we?!)
    """
    assert len(feature_data) == len(pixels)
    feature_target_sizes = feature_target_sizes or get_largest_features(feature_data)
    resized_vectors = []
    for index, feature in enumerate(feature_data):
        if feature.any():
            resized_vector = _get_resized_feature_vector(feature,
                                                         feature_target_sizes,
                                                         pixels[index],
                                                         maintain_aspect_ratio)
            resized_vectors.append(resized_vector)
        else:
            resized_vectors.append(None)
    return resized_vectors, feature_target_sizes


def _concatenate_vector(feature_vector):
    """
    Takes in a dict of feature vectors (nominally provided by a single element
    of `get_resized_feature_vectors`) and puts them in to one concatenated
    vector
    :param feature_vectors: dict of feature vectors for one image
    :return: vector with all the features concatenated in to one vector.
    """
    try:
        return np.concatenate(
            [feature.flatten() for feature in feature_vector.values()]
        )
    except AttributeError:
        return None


def get_concatenated_vectors(normalized_facial_vectors):
    """
    The bread and butter of this entire module. This method takes in the list
    of dictionaries of each face, concatenates each face's features together
    in to a single vector and puts it all together in to a
    num_faces * num_pixels_per_face array (with exception of Nones for images
    which could not be processed for face.)

    :param normalized_facial_vectors: the list (per face) of
    dictionaries (per facial feature) of normalized features
    :return: an almost-matrix of all your relevant datas. pixel data where
    available, None where no face was detected. (BEWARE!)
    ALSO: you will most likely need to put the result through np.vstack after
    any filtering you've done so as to have a usable format by scikitlearn and
    other libs (array of array doesn't do so well).
    """
    concatenation_list = []  # numpy sucks, so we do this in python.
    # do you even concatenate, bro?
    for face in normalized_facial_vectors:
        concatenation_list.append(_concatenate_vector(face))
    concatenated_faces = np.array(concatenation_list)
    return concatenated_faces
