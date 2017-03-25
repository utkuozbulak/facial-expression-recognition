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


def get_resized_feature_vectors(feature_data,
                                pixels,
                                maintain_aspect_ratio=False):
    assert len(feature_data) == len(pixels)
    feature_target_sizes = get_largest_features(feature_data)
    resized_vectors = []
    for index, feature in enumerate(feature_data):
        if feature.any():
            resized_vector = _get_resized_feature_vector(feature,
                                                         feature_target_sizes,
                                                         pixels[index],
                                                         maintain_aspect_ratio)
            resized_vectors.append(resized_vector)
            print(resized_vector)
        else:
            resized_vectors.append(None)
    return resized_vectors


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
    possible feature, this size is obtained from `get_largest_features`
    """
    resized_image_features = {}
    for feature_key in FACIAL_FEATURE_SETS.keys():
        # grab the feature out of the full image. how hard could it be?
        minx_index, maxx_index, miny_index, maxy_index = _get_min_max_feature_numbers(
            FACIAL_FEATURE_SETS[feature_key],
            feature_data
        )

        # now grab out the x/y's of the pixels - separated for some clarity.
        minx = int(feature_data[minx_index, 0])
        maxx = int(feature_data[maxx_index, 0])
        miny = int(feature_data[miny_index, 1])
        maxy = int(feature_data[maxy_index, 1])

        # ok, we have our data for this facial feature,
        # (scale and index of min/max points on each axis),
        # let's extract that part of the image and get it rescaled.
        # this is simple, use the min/max values
        feature_pixels = pixels[minx:maxx, miny:maxy]

        # kk resize dis up to our feature_target_size.
        target_size = (feature_target_size[feature_key]['x'],
                       feature_target_size[feature_key]['y'])
        try:
            resized_feature = imresize(feature_pixels, target_size)
        except ValueError as e:
            print("well, damn.")
        resized_image_features[feature_key] = resized_feature

    return resized_image_features
