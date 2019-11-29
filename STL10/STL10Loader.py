import numpy as np
import compress_pickle
import re

import os
import sys
sys.path.append(os.path.join("STL10"))

import stl10_input


class STL10Loader(object):

  class_names = ("airplane",
                  "bird",
                  "car",
                  "cat",
                  "deer",
                  "dog",
                  "horse",
                  "monkey",
                  "ship",
                  "truck")

  def __init__(self):
    # Replace "train" with "test" or "unlabeled"
    train_re = re.compile(r"train_")
    train_to_test = lambda x: train_re.sub("test_", x)
    current_dir = os.getcwd()
    os.chdir(os.path.join("STL10"))
    stl10_input.download_and_extract()
    # Populate the training set
    self.x_train = stl10_input.read_all_images(stl10_input.DATA_PATH)
    self.y_train = stl10_input.read_labels(stl10_input.LABEL_PATH)
    self.y_train -= 1  # Labels are not 0-indexed
    self.x_test = stl10_input.read_all_images(train_to_test(stl10_input.DATA_PATH))
    self.y_test = stl10_input.read_labels(train_to_test(stl10_input.LABEL_PATH))
    self.y_test -= 1  # Labels are not 0-indexed
    #self.x_unlabeled = stl10_input.read_all_images(train_to_unlabeled(stl10_input.DATA_PATH))
    os.chdir(current_dir)

  def _flatten_images(self, x):
    x = x.reshape(x.shape[0], np.prod(x.shape[1:])).astype(float)
    x *= 1./255.
    return x.astype(np.float32)

  def data(self, flattened=False, category_filter=None):
    """Extract data from the set"""
    if category_filter is not None:
      self.category_filter = category_filter
      filtered_indexes_train = self._filter_by_categories(self.y_train, category_filter)
      filtered_indexes_test = self._filter_by_categories(self.y_test, category_filter)
      x_train_local = self.x_train[filtered_indexes_train]
      y_train_local = self._reduce_filtered_y(self.y_train[filtered_indexes_train])
      x_test_local = self.x_test[filtered_indexes_test]
      y_test_local = self._reduce_filtered_y(self.y_test[filtered_indexes_test])
    else:
      x_train_local = self.x_train
      y_train_local = self.y_train
      x_test_local = self.x_test
      y_test_local = self.y_test
    if flattened:
      return ((self._flatten_images(x_train_local),
               y_train_local),
              (self._flatten_images(x_test_local),
               y_test_local))
    return ((x_train_local, y_train_local), (x_test_local, y_test_local))

  def _filter_by_categories(self, y, category_filter):
    """Returns indexes of filtered examples by categories given"""
    reduced_categories = [self.class_names.index(n) for n in category_filter]
    return np.where([n in reduced_categories for n in y])[0]

  def _reduce_filtered_y(self, y):
    y_uniquelist = list(set(y))
    y_reduced = [n[0] for n in enumerate(y_uniquelist)]
    reduction_dict = {n[0]: n[1] for n in zip(y_uniquelist, y_reduced)}
    return np.array([reduction_dict[n] for n in y])

  def get_reduced_class_names(self):
    try:
      return tuple(filter(self.class_names, lambda n: n in self.category_filter))
    except AttributeError:
      return self.class_names

if __name__ == "__main__":
  stl10_loader = STL10Loader()
  print(len(stl10_loader.x_train))
  print(len(stl10_loader.x_test))
  pickle_filename = "stl10_dataset.pickle.gz"
  print("Pickling the loader to %s..." % pickle_filename)
  compress_pickle.dump(stl10_loader, pickle_filename)
  print("Done.")
