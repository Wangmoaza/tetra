from tetra import *
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt


##### build autoencoder model #####

input_data = Input(shape=(256, )) # input placeholder
# encoded representation of the input
encoded = Dense(128, activation='relu')(input_data)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(2, activation='relu')(encoded)

encoder = Model(input=input_data, outpout=encoded)
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

##### get input #####


tetra_freq = np.load('tetrafreq.npy')
phylum_index = np.load('phylumIndex.npy')
phylum_names = np.load('phylumNames.npy')

tetra_train, tetra_test = train_test_split(tetra_freq, test_size=0.1,
  random_state=42)

print "train set shape: ", tetra_train.shape
print "test set shape: ", tetra_test.shape

##### train the encoder model #####

encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder.fit(tetra_train, tetra_train, nb_epoch=50, batch_size =256,
  shuffle=True, validation_data=(tetra_test, tetra_test))

encoded_result = encoder.predict(tetra_test)
print encoded_result
