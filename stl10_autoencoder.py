import numpy as np 
import compress_pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from StackedAutoencoder import *
from STL10Loader import *


def autoencoder_train(dataset="stl10_dataset.pickle.gz"):
    stl10_dataset = compress_pickle.load(dataset)
    ((x_train, y_train), (x_test, y_test)) = stl10_dataset.data(flattened=True)
    # Create and train stacked autoencoders
    s = StackedAutoencoderTrain()
    print("Training first layer...")
    s.train_first_layer(64, x_train, y_train, x_test, y_test)
    print("Training second layer...")
    s.train_second_layer(32)
    print("Training classifier...")
    s.train_classifier(stl10_dataset.class_names)
    print("Done.")
    return s

if __name__ == "__main__":
    autoencoder_train()
