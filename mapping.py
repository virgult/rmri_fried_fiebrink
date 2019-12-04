import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


class Mapping(object):
    def __init__(self, gtzan, stl10):
        self.gtzan = gtzan
        self.stl10 = stl10
        self.gtzan_l2_norms = self.compute_l2norms(self.gtzan['x'])
        self.stl10_l2_norms = self.compute_l2norms(self.stl10['x'])
        self.mapping = self.map_stl10_to_gtzan()

    def display_stl10_image(self, example_index):
        """Plots image from STL10 dataset"""
        matrix = self.stl10[example_index].reshape(96, 96, 3)
        plt.imshow(matrix)
        plt.show()

    def compute_l2norms(self, vectors):
        l2norms = np.zeros(len(vectors))
        for index, vector in enumerate(vectors):
            l2norms[index] = LA.norm(vector)

        return l2norms

    def map_stl10_to_gtzan(self):
        mapping_array = np.zeros((len(self.stl10_l2_norms), 2))
        for index, gtzan_norm in enumerate(self.gtzan_l2_norms):
            smallest_dist = np.inf
            for stl10_index, stl10_norm in enumerate(self.stl10_l2_norms):
                distance = abs(stl10_norm - gtzan_norm)
                if distance < smallest_dist:
                    mapping_array[index][0] = distance
                    mapping_array[index][1] = stl10_index
                    smallest_dist = distance
                    if index == 7:
                        print(smallest_dist)
        return mapping_array

    def map_image_to_sound(self, image_index):
        return self.mapping[image_index]