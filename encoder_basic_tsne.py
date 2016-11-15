### use basic autoencoder to reduce 256 -> 32
### use t-sne to reduce 32 -> 2
### divide training data into two groups
from keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input
from keras.utils import np_utils
import numpy as np
from sklearn.manifold import TSNE

# load data
tetra_freq = np.load('tetrafreq.npy')
phylum_index = np.load('phylumIndex.npy')
phylum_names = np.load('phylumNames.npy')

#X_odd = tetra_freq[1::2, :]
#X_even = tetra_freq[0::2, :]

encoding_dim = 32
input_data = Input(shape=(256,))

# layers
encoded = Dense(encoding_dim, activation='relu')(input_data)
decoded = Dense(256, activation='relu')(encoded)

autoencoder = Model(input=input_data, output=decoded)
encoder = Model(input=input_data, output=encoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(tetra_freq, tetra_freq, batch_size=256, nb_epoch=100, shuffle=True)

encoded_result = encoder.predict(tetra_freq)
encoded_odd = encoded_result[1::2, :]
encoded_even = encoded_result[0::2, :]

# use T-SNE to reduce 32-d to 2-d

tsne = TSNE(n_components=2, random_state=0)
tsne_result_odd = tsne.fit_transform(encoded_odd)
tsne_result_even = tsne.fit_transform(encoded_even)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure()
for c, i, name in zip ("bgrcmykw", list(range(7, -1, -1)), phylum_names):
    plt.scatter(tsne_result_[phylum_index == i, 0], tsne_result[phylum_index == i, 1], c=c, label=name)
    plt.scatter(tsne_result[phylum_index == i, 0], tsne_result[phylum_index == i, 1], c=c, label=name)

#np.set_printoptions(suppress=True)
plt.title('Encoded result of tetranucleotide')
plt.legend(loc=3, fontsize=10)
plt.savefig('basic_encoder_tsne.png')
 
