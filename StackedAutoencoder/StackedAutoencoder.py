import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

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
    self.model.layers[index].trainable = False;
    
  def defreeze_layer(self, index):
    self.model.layers[index].trainable = True;
    
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
    model.save_weights(weights_h5_filename)


class StackedAutoencoderTrain(object):
  """Stacked Autoencoder training class"""

  def train_first_layer(self, num_units, x_train, y_train, x_test, y_test,
                        n_epochs=50, learning_rate=1.):
    """Creates and trains first layer"""
    input_dim = x_train.shape[1]
    # Create 1st Autoencoder
    self.autoencoder_1 = DeepAutoEncoder(n_layers=1, 
                                         units=num_units, 
                                         input_dim=input_dim)
    # Optimiser
    optimiser = Adadelta(lr=learning_rate)
    # Compile Model
    self.autoencoder_1.model.compile(optimizer=optimiser, loss='binary_crossentropy')
    print("Learning rate: %s" % K.eval(self.autoencoder_1.model.optimizer.lr))
    self.autoencoder_1.model.fit(x_train, x_train,
                                 epochs=n_epochs,
                                 batch_size=256,
                                 shuffle=True,
                                 validation_data=(x_test, x_test))
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.y_test = y_test
    self.num_units = num_units
    self.input_dim = input_dim
    
  def train_second_layer(self, n_units_2, n_epochs=50):
    """Creates and trains inner layer"""
    self.num_units_2 = n_units_2
    self.autoencoder_2 = DeepAutoEncoder(n_layers=2, 
                                         units=[self.num_units, self.num_units_2], 
                                         input_dim=self.input_dim)
    # Set weights of 1st layer
    self.autoencoder_2.set_layer_weights(1, self.autoencoder_1.model.layers[1].get_weights())
    # Freeze 1st layer
    self.autoencoder_2.freeze_layer(1)
    # Compile model
    self.autoencoder_2.model.compile(optimizer='adadelta', loss='binary_crossentropy')
    print("Learning rate: %s" % K.eval(self.autoencoder_2.model.optimizer.lr))
    # Train on data
    self.autoencoder_2.model.fit(self.x_train, self.x_train,
                                 epochs=n_epochs,
                                 batch_size=256,
                                 shuffle=True,
                                 validation_data=(self.x_test, self.x_test))
    # Freeze 2nd layer
    self.autoencoder_2.freeze_layer(2)

  def train_classifier(self, classes_vector, n_epochs=50):
    """Creates and trains end classifier"""
    self.n_classes = len(classes_vector)
    self.classes_vector = classes_vector
    self.y_train_encoded = to_categorical(self.y_train, self.n_classes)
    self.y_test_encoded = to_categorical(self.y_test, self.n_classes)
    # Create classifier
    classifier_output = Dense(self.n_classes, activation='softmax')(self.autoencoder_2.encoder_layers[1])
    self.classifier = Model(self.autoencoder_2.input,classifier_output)
    self.classifier.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
    print("Learning rate: %s" % K.eval(self.classifier.optimizer.lr))
    self.classifier.fit(self.x_train,self.y_train_encoded,
                        validation_data=(self.x_test,self.y_test_encoded),
                        epochs=n_epochs,
                        batch_size=32)
  
  def predict(self, example):
    raise NotImplementedError

  def calculate_accuracy(self):
    raise NotImplementedError


class DeepAutoencoderTrain(object):
  """Deep Autoencoder training boilerplate"""

  def train_autoencoder(self, num_units, x_train, y_train, x_test, y_test,
                        n_epochs=50, learning_rate=1.):
    """Creates and trains deep autoencoder"""
    # Declare Deep AutoEncoder
    self.num_layers = len(num_units)
    input_dim = x_train.shape[1]
    self.deep_autoencoder = DeepAutoEncoder(n_layers=self.num_layers, 
                                            units=num_units, 
                                            input_dim=input_dim)
    # Compile Model
    self.deep_autoencoder.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
    # Train and save history
    self.model_history = self.deep_autoencoder.fit(x_train,
        x_train,
        epochs=n_epochs,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test))
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.y_test = y_test
    self.num_units = num_units
    self.input_dim = input_dim

  def train_classifier(self, classes_vector, n_epochs=50):
    """Creates and trains end classifier"""
    self.n_classes = len(classes_vector)
    self.classes_vector = classes_vector
    self.y_train_encoded = to_categorical(self.y_train, self.n_classes)
    self.y_test_encoded = to_categorical(self.y_test, self.n_classes)
    # Freeze autoencoder layers
    #self.deep_autoencoder.freeze_layer(1)
    #self.deep_autoencoder.freeze_layer(2)
    # Create classifier
    classifier_output = Dense(self.n_classes, activation='softmax')(self.deep_autoencoder.encoded)
    self.classifier = Model(self.deep_autoencoder.input, classifier_output)
    self.classifier.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
    self.classifier.summary()
    time.sleep(3.)
    print("Learning rate: %s" % K.eval(self.classifier.optimizer.lr))
    self.model_history = self.classifier.fit(self.x_train,self.y_train_encoded,
                        validation_data=(self.x_test,self.y_test_encoded),
                        epochs=n_epochs,
                        batch_size=32)

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
    json_filename = "%s_model.json" % now_string()
    h5_filename = "%s_weights.h5" % now_string()
    self.deep_autoencoder.save_model(json_filename, h5_filename)


class Conv2DDeepAutoEncoderTrain(object):

  def train_autoencoder(self, num_units, hidden_units, x_train, y_train, x_test, y_test,
                        n_epochs=50):
    #ENCODER
    input_shape = x_train.shape[1:]
    inp = Input(input_shape)
    e = Conv2D(num_units[0], (3, 3), activation='relu')(inp)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(num_units[1], (3, 3), activation='relu')(e)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(num_units[2], (3, 3), activation='relu')(e)
    l = Flatten()(e)
    l = Dense(np.prod(hidden_units), activation='softmax')(l)
    #DECODER
    d = Reshape(hidden_units)(l)
    d = Conv2DTranspose(num_units[2],(3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(num_units[1],(3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(num_units[0],(3, 3), activation='relu', padding='same')(d)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)
    ae = Model(inp, decoded)
    ae.summary()
