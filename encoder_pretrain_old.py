from keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np

# load dataset
X = np.load('tetrafreq.npy') # shape (57681, 256)
phylum_index = np.load('phylumIndex.npy')
phylum_names = np.load('phylumNames.npy')

pre_trained_layers = []
encoders = []
decoders = []
nb_hidden_layers = [X.shape[1], 64, 16, 2]
X_tmp = np.copy(X)

np.random.seed(3)
nb_epoch_pretraining = 10
batch_size_pretraining = 500


for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], 
                                  nb_hidden_layers[1:]), start=1):
	print('Training the layer {0}: input {1} -> output {2}'.format(i, n_in, n_out))
	# create AE and training
	ae = Sequential()
	input_data = Input(shape=(n_in, ))
	#encoded = Dense(output_dim=n_out, input_dim=n_in, activation='tanh')(input_data)
	encoder = Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='relu')])
        #encoded = encoder(input_data)
	
        #decoded = Dense(output_dim=n_in, input_dim=n_out, activation='tanh')(encoded)
	decoder = Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='relu')])
	#decoded = decoder(encoded)

        #ae.add(Sequential([encoder, decoder]))
        #encoder = Model(input=input_data, output=encoded)
	ae.add(encoder)
        ae.add(decoder)
        #sgd = SGD(lr=2, decay=1e-6, momentum=0.0, nesterov=True)
	ae.compile(loss='mse', optimizer='adam')
	ae.fit(X_tmp, X_tmp, batch_size=batch_size_pretraining,
		   nb_epoch=nb_epoch_pretraining, verbose=True, shuffle=True)

	# store trained weight and update training data
	#encoders.append(ae.layers[0])
	#decoders.append(ae.layers[1])
        pre_trained_layers.append((ae.layers[0], ae.layers[0].get_weights()))
        ae.summary()
	X_tmp = encoder.predict(X_tmp)
### END - for


# fine-tuning
print 'fine-tuning'

model = Sequential()
encoder_model = Sequential()
        
for layer, weights in pre_trained_layers:
        encoder_model.add(layer)
        encoder_model.layers[-1].set_weights(weights)

model.add(encoder_model)
# Add the ouput layer
model.add(Dense(nb_hidden_layers[0], input_dim=nb_hidden_layers[-1], activation='relu'))

# train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X, X, batch_size=batch_size_pretraining, nb_epoch=50)
encoded_result = encoder_model.predict(X)
"""
# End to End autoencoder training

if len(nb_hidden_layers) > 2:
    full_encoder = Sequential()
    for encoder in encoders:
        full_encoder.add(encoder)

    full_decoder = Sequential()
    for decoder in reversed(decoders):
        full_decoder.add(decoder)
    
    full_ae = Sequential()
    full_ae.add([full_encoder, full_decoder])
    full_ae.compile(loss='mse', optimizer='adam')

    print "Pretraining of full AE"
    full_ae.fit(X, X, batch_size=batch_size_pretraining, nb_epoch=50, shuffle=True)
    encoded_result = full_encoder.predict(X)
"""
# plot the encoded result

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    

plt.figure()
for c, i, name in zip ("bgrcmykw", list(range(7, -1, -1)), phylum_names):
    plt.scatter(encoded_result[phylum_index == i, 0], encoded_result[phylum_index == i, 1], c=c, label=name)
plt.title('Encoded result of tetranucleotide (with pretraining)')
plt.legend(loc=3, fontsize=10)
plt.savefig('encoder_pre.png')

