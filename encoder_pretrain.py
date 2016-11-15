import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input
from keras.utils import np_utils
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.manifold import TSNE
from keras import regularizers
from keras.callbacks import TensorBoard

def load_data(tetraFile, taxonFile, rank, minCnt=10):
    X = np.load(tetraFile)
    taxons = np.load(taxonFile)
    names = getTopNames(taxons, rank, minCnt)
    y = label2num(taxons, names, rank) 
    return X, y, names
### END - def load_data


def getTopNames(taxons, rank, minCnt):
    # make dictionary {phylumName: count}
    dic = {}
    for item in taxons[:, rank]:
        dic[item] = dic.get(item, 0) + 1

    names = []
    for label in dic.keys():
        if dic.get(label) >= minCnt:
            names.append(label)
            print (label, dic.get(label))
    ### END - for label
    names.append('Others')
    
    print names
    return names
### END - getTopNames


def label2num(taxons, names, rank):
    # make list of phylum names in index 0: Bacteriodetes, etc.
    labels = taxons[:, rank]
    indexList = []
    
    for i in range(len(labels)):
        item = labels[i]
        if item in names:
            indexList.append(names.index(item))
        else:
            indexList.append(len(names)-1) # last element is names is 'Others'
    ### END - for i
        
    for i in range(len(names)):
        print i, indexList.count(i)
    
    return np.asarray(indexList)
### END - def label2num

def pretrain(X):
    encoder_weights = []
    decoder_weights = []
    nb_hidden_layers = [X.shape[1], 500, 100, 20, 2]
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
        encoder = Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='relu')])
        decoder = Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='relu')])

        ae.add(encoder)
        ae.add(decoder)
        ae.compile(loss='mse', optimizer='adam')
        ae.fit(X_tmp, X_tmp, batch_size=batch_size_pretraining,
               nb_epoch=nb_epoch_pretraining, verbose=True, shuffle=True)

        ae.summary()
        # store trained weight and update training data
        encoder_weights.append(ae.layers[0].get_weights())
        decoder_weights.append(ae.layers[1].get_weights())

        X_tmp = encoder.predict(X_tmp)
    ### END - for
    
    return encoder_weights, decoder_weights
### END - def pretrain

def finetune(X, y, encoder_weights, decoder_weights, kfold=True):
    n_in = X.shape[1]
    input_data = Input((n_in, ))

    # encoder
    encoded = Dense(500, activation='relu', weights=encoder_weights[0])(input_data)
    encoded = Dense(100, activation='relu', weights=encoder_weights[1])(encoded)
    encoded = Dense(20, activation='relu', weights=encoder_weights[2])(encoded)
    encoded = Dense(2, activation='relu', weights=encoder_weights[3])(encoded)

    # decoder
    decoded = Dense(20, activation='relu', weights=decoder_weights[3])(encoded)
    decoded = Dense(100, activation='relu', weights=decoder_weights[2])(decoded)
    decoded = Dense(500, activation='relu', weights=decoder_weights[1])(decoded)
    decoded = Dense(n_in, weights=decoder_weights[0])(decoded)


    ae = Model(input=input_data, output=decoded)
    encoder = Model(input=input_data, output=encoded)
    ae.compile(loss='mse', optimizer='rmsprop')
    
    # train using stratified kfold
    if kfold == True:
        skf = StratifiedKFold(y, n_folds = 5) # FIXME
        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            ae.fit(X_train, X_train, 
                            nb_epoch=50, 
                            batch_size=256, 
                            shuffle=True, 
                            validation_data=(X_test, X_test))

    # train using all data
    else:
        # CHANGED: try validation_split param
        ae.fit(X, X, nb_epoch=200, batch_size=256, shuffle=True,
                validation_split=0.1)

    encoded_result = encoder.predict(X)
    
    print "encoded result shape", encoded_result.shape
    print encoded_result

    return encoded_result
### END - finetune


def tsne(X):
    tsne = TSNE(learning_rate=1000, n_components=2, verbose=2)
    tsne_result = tsne.fit_transform(X)
    
    return tsne_result
### END - tsne


def plot(result, y, names, title, figName):
    plt.figure()

    for c, i, name in zip("bgrcmykw", list(range(0, len(names))), names):
        plt.scatter(result[y==i, 0], result[y==i, 1], c=c, label=name)
    plt.legend(loc=0, fontsize=10)
    plt.title(title)
    plt.savefig(figName)
### END - plot


def main():
    phylum_names = np.load('db2_phylum_names.npy')
    """
    for phy in phylum_names[:-1]:
        print '******** ' + phy + '********'
        X, y, names = load_data(tetraFile='db2_tetra_phylum_' + phy +'.npy', 
                                taxonFile='db2_taxon_phylum_' + phy + '.npy', 
                                rank=2, minCnt=30)
        encoder_weights, decoder_weights = pretrain(X)
        result = finetune(X, y, encoder_weights, decoder_weights, kfold=False)
        title = "Encoded TNA of " + phy
        figName = 'encoder_256-500-100-20-10-2_' + phy + '.png'
        plot(result, y, names, title, figName)
    """
    X, y, names = load_data(tetraFile='db2_tetra_phylum_' + 'Firmicutes' +'.npy', 
                                taxonFile='db2_taxon_phylum_' + "Firmicutes" + '.npy', 
                                rank=2, minCnt=30)
    encoder_weights, decoder_weights = pretrain(X)
    result = finetune(X, y, encoder_weights, decoder_weights, kfold=False)
    

main()
