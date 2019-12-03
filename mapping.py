import numpy as np
from numpy import linalg as LA
from StackedAutoencoder import StackedAutoencoder


#todo change paths below
GTZAN_PATH = ""
STL10_PATH = ""
GTZAN_WEIGHTS_PATH = ""


def compute_l2norms(vectors):
    l2norms = np.zeros(len(vectors))

    for index, vector in enumerate(vectors):
        l2norms[index] = LA.norm(vector)

    return l2norms


def map_stl10_to_gtzan(l2_norms_stl10, l2_norms_gtzan):
    mapping = np.zeros(2, len(l2_norms_stl10))
    for index, gtzan_norm in enumerate(l2_norms_stl10):
        closest_v = np.inf
        for norm in l2_norms_gtzan:
            distance = abs(norm - gtzan_norm)
            if distance < closest_v:
                mapping[0][index] = distance
                mapping[1][index] = index
    return mapping
