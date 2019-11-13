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

  def data(self, flattened=False):
    """Extract data from the set"""
    if flattened:
      return ((self._flatten_images(self.x_train),
               self.y_train),
              (self._flatten_images(self.x_test),
               self.y_test))
    return ((self.x_train, self.y_train), (self.x_test, self.y_test))

if __name__ == "__main__":
  stl10_loader = STL10Loader()
  print(len(stl10_loader.x_train))
  print(len(stl10_loader.x_test))
  pickle_filename = "stl10_dataset.pickle.gz"
  print("Pickling the loader to %s..." % pickle_filename)
  compress_pickle.dump(stl10_loader, pickle_filename)
  print("Done.")
