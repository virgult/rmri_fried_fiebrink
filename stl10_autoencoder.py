import numpy as np 
import compress_pickle
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from StackedAutoencoder.StackedAutoencoder import DeepAutoencoderTrain
from STL10.STL10Loader import STL10Loader


def deep_autoencoder_train(dataset="stl10_dataset.pickle.gz",
                           layer_dims=[256, 64],
                           batch_size=96,
                           autoencoder_epochs=5,
                           classifier_epochs=5):
    stl10_dataset = compress_pickle.load(dataset)
    category_filter = ("airplane", "car", "cat", "dog")
    ((x_test, y_test), (x_train, y_train)) = stl10_dataset.data(flattened=True,
      category_filter=category_filter)
    print("TRAIN/TEST SET DIMENSIONS")
    print("Training set: %s" % y_train.shape[0])
    print("Test set: %s" % y_test.shape[0])
    time.sleep(3.)
    # Create and train stacked autoencoders
    s = DeepAutoencoderTrain()
    print("Training autoencoder...")
    try:
        auto_batch_size = batch_size[1]
    except TypeError:
        auto_batch_size = batch_size
    s.train_autoencoder(layer_dims, x_train, y_train, x_test, y_test,
                        batch_size=auto_batch_size,
                        n_epochs=autoencoder_epochs)
    print("Training classifier...")
    try:
        classif_batch_size = batch_size[0]
    except TypeError:
        classif_batch_size = batch_size
    s.train_classifier(stl10_dataset.get_reduced_class_names(),
                       batch_size=classif_batch_size,
                       n_epochs=classifier_epochs)
    print("Plotting results...")
    s.plot_model_performance()
    print("Predicting encoded examples...")
    s.save_model()
    print("Saving model...")
    s.dump_predicted_set(x_test, y_test)
    print("Done.")
    return s


if __name__ == "__main__":
    deep_autoencoder_train(dataset="stl10_dataset.pickle.gz", autoencoder_epochs=20, classifier_epochs=25, batch_size=96)
