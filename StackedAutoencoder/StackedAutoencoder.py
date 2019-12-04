import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import compress_pickle

try:
  # Install the plaidml backend
  import plaidml.keras
  plaidml.keras.install_backend()
except:
  # Install the tensorflow backend
  import tensorflow as tf


from keras.layers import Input, Dense, Dense, Conv2D, Dropout, \
  BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, \
  Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adadelta
from keras import backend as K

class DeepAutoEncoder(object):
  """Stacked Autoencoder Topology (generic)"""

  def __init__(self, n_layers, units, input_dim, activation='relu'):
    self.n_layers = n_layers if (n_layers > 1) else 1
    self.set_units(units)
    self.activation = activation
    self.input_dim = input_dim
    try:
      # Convolutional case
      iter(self.input_dim)
      self.input = Input(shape=self.input_dim)
    except TypeError:
      # Other cases
      self.input = Input(shape=(self.input_dim,))
    self.model = Model(self.input, self.decoder(self.encoder(self.input)))
    self.encoder_model = Model(self.input, self.encoded)

  def set_units(self, units):
    try:
      iter(units)
      if len(units) != self.n_layers:
         raise RuntimeError("List of units doesn't match the number of layers.")
      self.units = units
    except TypeError:
      self.units = [units] * self.n_layers
      
  def encoder(self, input):
    self.encoder_layers = []
    encoded = Dense(self.units[0], activation=self.activation)(input)
    self.encoder_layers.append(encoded)
    if self.n_layers > 1:
      for e in range(1, self.n_layers):
        encoded =  Dense(self.units[e], activation=self.activation)(encoded)
        self.encoder_layers.append(encoded)
    self.encoded = encoded
    return encoded
  
  def decoder(self, encoded):
    self.decoder_layers = []
    decoded = encoded
    self.decoder_layers.append(decoded)
    if self.n_layers > 1:
      for e in range(self.n_layers-1, 0, -1):
        decoded = Dense(self.units[e], activation=self.activation)(decoded)
        self.decoder_layers.append(decoded)
    decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
    return decoded

  def compile(self, *args, **kwargs):
    return self.model.compile(*args, **kwargs)

  def fit(self, *args, **kwargs):
    return self.model.fit(*args, **kwargs)

  def freeze_layer(self, index):
    self.model.layers[index].trainable = False
    
  def defreeze_layer(self, index):
    self.model.layers[index].trainable = True
    
  def set_layer_weights(self, index, weights):
    self.model.layers[index].set_weights(weights)

  def get_layer_weights(self, index):
    return self.model.layers[index].get_weights()

  def save_model(self, model_json_filename, weights_h5_filename):
    # serialize model to JSON
    model_json = self.model.to_json()
    with open(model_json_filename, "w") as fh:
        fh.write(model_json)
    # serialize weights to HDF5
    self.model.save_weights(weights_h5_filename)

  def predict_encoded(self, x_set):
    predict_model = Sequential(self.model.layers[:len(self.model.layers)//2+1])
    predict_model.compile(optimizer="adadelta",
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    # Prevent training
    for n in range(len(predict_model.layers)):
      predict_model.layers[n].trainable = False
    # Predict all Y's at once
    return predict_model.predict(x_set)

class DeepAutoencoderTrain(object):
  """Deep Autoencoder training boilerplate"""

  def train_autoencoder(self, num_units, x_train, y_train, x_test, y_test,
                        n_epochs=50, learning_rate=1., batch_size=32,
                        optimizer='adam', loss_function='binary_crossentropy'):
    """Creates and trains deep autoencoder"""
    # Declare Deep AutoEncoder
    self.num_layers = len(num_units)
    input_dim = x_train.shape[1]
    self.deep_autoencoder = DeepAutoEncoder(n_layers=self.num_layers, 
                                            units=num_units, 
                                            input_dim=input_dim)
    # Compile Model
    self.deep_autoencoder.compile(optimizer=optimizer,
                             loss=loss_function,
                             metrics=['accuracy'])
    # Train and save history
    self.model_history = self.deep_autoencoder.fit(x_train,
        x_train,
        epochs=n_epochs,
        batch_size=batch_size,
        shuffle=True,
        #validation_data=(x_test, x_test))
        validation_split=0.1)
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.y_test = y_test
    self.num_units = num_units
    self.input_dim = input_dim

  def train_classifier(self, classes_vector, n_epochs=50, batch_size=32, encoded=False,
                        optimizer='adam', loss_function='binary_crossentropy'):
    """Creates and trains end classifier"""
    self.n_classes = len(classes_vector)
    self.classes_vector = classes_vector
    if not encoded:
        self.y_train_encoded = to_categorical(self.y_train, self.n_classes)
        self.y_test_encoded = to_categorical(self.y_test, self.n_classes)
    else:
        self.y_train_encoded = self.y_train
        self.y_test_encoded = self.y_test
    # Create classifier
    classifier_output = Dense(self.n_classes, activation='softmax')(self.deep_autoencoder.encoded)
    self.classifier = Model(self.deep_autoencoder.input, classifier_output)
    # Freeze autoencoder layers
    #for n in range(1, self.num_layers+1):
    #  self.classifier.layers[n].trainable = False
    self.classifier.compile(optimizer=optimizer,loss=loss_function,metrics=['accuracy'])
    self.classifier.summary()
    time.sleep(3.)
    print("Learning rate: %s" % K.eval(self.classifier.optimizer.lr))
    self.model_history = self.classifier.fit(self.x_train,self.y_train_encoded,
                        #validation_data=(self.x_test,self.y_test_encoded),
                        validation_split=0.1,
                        epochs=n_epochs,
                        batch_size=batch_size)
    loss, acc = self.classifier.evaluate(self.x_test, self.y_test_encoded, batch_size=batch_size)
    print("Trained classifier:\nLOSS: %s\nACCURACY:%s" % (loss, acc))

  def plot_model_performance(self):
    now_string = self._get_now_string()
    # Plot training & validation accuracy values
    plt.plot(self.model_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("%s_accuracy.png" % now_string)
    # Plot training & validation loss values
    plt.figure()
    plt.plot(self.model_history.history['loss'])
    plt.plot(self.model_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("%s_loss.png" % now_string)

  def _get_now_string(self):
    try:
      now_string = self.now_string
    except AttributeError:
      now = datetime.datetime.now()
      now_string = "%s%s%s_%s%s%s" % (now.year,
                                      now.month,
                                      now.day,
                                      now.hour,
                                      now.minute,
                                      now.second)
      self.now_string = now_string
    return now_string

  def save_model(self):
    now_string = self._get_now_string()
    json_filename = "%s_model.json" % now_string
    h5_filename = "%s_weights.h5" % now_string
    self.deep_autoencoder.save_model(json_filename, h5_filename)

  def dump_predicted_set(self, x_set, y_set):
    x_set_enc = self.deep_autoencoder.predict_encoded(x_set)
    dump = {
      "categories": self.classes_vector,
      "x" : x_set,
      "x_encoded": x_set_enc,
      "y" : y_set
    }
    now_string = self._get_now_string()
    dump_filename = "%s_encoded.gz" % now_string
    compress_pickle.dump(dump, dump_filename)

