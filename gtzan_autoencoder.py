import numpy as np
from keras.models import Model
from keras.layers import Dense
from StackedAutoencoder import StackedAutoencoder


features = np.load("./data/flatten_mfcc.npy")
labels = np.load("./data/one_hot_labels.npy")

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
deep_autoencoder = StackedAutoencoder.DeepAutoEncoder(n_layers=num_layers,
                                units=units,
                                input_dim=input_dim)
# Compile Model
deep_autoencoder.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
deep_autoencoder.model.summary()
deep_autoencoder.encoder_model.summary()

# Train Deep AutoEncoder
deep_autoencoder_history = deep_autoencoder.model.fit(train_input, train_input,
                epochs=training_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.1)

n_classes = 10
classifier_output = Dense(n_classes,activation='softmax')(deep_autoencoder.encoded)
classifier = Model(deep_autoencoder.input,classifier_output)
classifier.summary()
for i,layer in enumerate(classifier.layers):
    print(i,layer.name,layer.trainable)

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(train_input, train_labels, epochs=30, batch_size=batch_size,
          validation_split=0.2)

loss, acc = classifier.evaluate(test_input, test_labels, batch_size=batch_size)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))
