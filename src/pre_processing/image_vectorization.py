
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

# vertical is up/down, horizontal is left/right
# FEATURE_AXIS_POINTS = {
#     "mouth": {
#         "vertical": (51, 57),
#         "horizontal": (48, 54),
#     },
#     "nose": {
#         "vertical": (27, 33),
#         "horizontal": (31, 35),
#     },
#     "left_eye": {
#         "vertical": (),
#         "horizontal": (36, 39),
#     },
#     "right_eye": {
#         "vertical": (),
#         "horizontal": (42, 45),
#     },
#     "left_brow": {
#         "vertical": (),
#         "horizontal": (42, 45),
#     },
#     "right_brow": {
#         "vertical": (),
#         "horizontal": (42, 45),
#     },
#
# }


def _get_largest_feature(feature_points, facial_vector):
    """
    get the maximum vertical and horizontal sizes of a particular feature,
    given a set of coordinates will simply calculate:
    max(x-axis) - min(x-axis)
    max(y-axis) - min(x_axis)
    :param feature_points: the particular points of the feature you're
    trying to find the size of, i.e: points 4,5,6
    :param facial_vector: the facial vector from which to get the min/max
    of these points.
    :return: the x and y sizes.
    """
    interesting_data_x = facial_vector[feature_points, 0]
    interesting_data_y = facial_vector[feature_points, 1]

    x_size = max(interesting_data_x) - min(interesting_data_x)
    y_size = max(interesting_data_y) - min(interesting_data_y)

    return x_size, y_size


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
                x_size, y_size = _get_largest_feature(
                    FACIAL_FEATURE_SETS[feature_key], face
                )
                max_feature_size[feature_key]['x'] = max(
                    x_size, max_feature_size[feature_key]['x']
                )
                max_feature_size[feature_key]['y'] = max(
                    y_size, max_feature_size[feature_key]['y']
                )
    return max_feature_size
