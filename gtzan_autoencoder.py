import numpy as np
from StackedAutoencoder import 


features = np.load("./flatten_mfcc.npy")
labels = np.load("./one_hot_labels.npy")

training_split = 0.5

# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

print(np.shape(train))
print(np.shape(test))

train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]

print(np.shape(train_input))
print(np.shape(train_labels))

# Define layer structure (units/layer)
units = [256, 64]
num_layers = len(units)
input_dim = np.shape(train_input)[1]

training_epochs = 20
batch_size = 32

# Declare Deep AutoEncoder
deep_autoencoder = AutoEncoder(n_layers=num_layers,
                                units=units,
                                input_dim=input_dim)
# Compile Model
deep_autoencoder.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
deep_autoencoder.model.summary()
deep_autoencoder.encoder_model.summary()

