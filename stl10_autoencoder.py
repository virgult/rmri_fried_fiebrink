import numpy as np 
import compress_pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from StackedAutoencoder.StackedAutoencoder import StackedAutoencoderTrain, DeepAutoencoderTrain,\
    Conv2DDeepAutoEncoderTrain
from STL10.STL10Loader import STL10Loader


def stacked_autoencoder_train(dataset="stl10_dataset.pickle.gz"):
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


def deep_autoencoder_train(dataset="stl10_dataset.pickle.gz"):
    stl10_dataset = compress_pickle.load(dataset)
    ((x_train, y_train), (x_test, y_test)) = stl10_dataset.data(flattened=True,
      category_filter=("airplane", "car", "cat", "dog"))
    # Create and train stacked autoencoders
    s = DeepAutoencoderTrain()
    print("Training autoencoder...")
    s.train_autoencoder([64, 16], x_train, y_train, x_test, y_test, n_epochs=25)
    print("Training classifier...")
    s.train_classifier(stl10_dataset.class_names, n_epochs=25)
    print("Done.")
    s.plot_model_performance()
    return s

def conv_autoencoder_train(dataset="stl10_dataset.pickle.gz"):
    stl10_dataset = compress_pickle.load(dataset)
    ((x_train, y_train), (x_test, y_test)) = stl10_dataset.data(flattened=False)
    s = Conv2DDeepAutoEncoderTrain()
    s.train_autoencoder((32, 64, 64), (6, 6, 1), x_train, y_train, x_test, y_test)
    

if __name__ == "__main__":
    deep_autoencoder_train(dataset="stl10_dataset.pickle.gz")

