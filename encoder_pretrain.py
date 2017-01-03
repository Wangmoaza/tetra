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
from sklearn import preprocessing
from keras import regularizers
from keras.callbacks import TensorBoard
from clustering import *

# load dataset
def load_data(tetraFile, taxonFile, rank, minCnt=10):
    X = np.load(tetraFile)
    taxons = np.load(taxonFile)
    # standardize data
    X_scaled = preprocessing.scale(X) + 1
    names = getTopNames(taxons, rank, minCnt)
    y = label2num(taxons, names, rank) 
    return X_scaled, y, names
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
            indexList.append(len(names)-1) # last element is 'Others'
    ### END - for i
        
    for i in range(len(names)):
        print i, indexList.count(i)
    
    return np.asarray(indexList)
### END - def label2num


def plot2d(result, y, names, title, figName):
    plt.figure()

    for c, i, name in zip("bgrcmykw", list(range(0, len(names))), names):
        plt.scatter(result[y==i, 0], result[y==i, 1], c=c, label=name)
    plt.legend(loc=0, fontsize=10)
    plt.title(title)
    plt.savefig(figName + '.png')
### END - plot2d


def pretrain(X):
    encoder_weights = []
    decoder_weights = []
    nb_hidden_layers = [X.shape[1], 500, 100, 20, 2]
    X_tmp = np.copy(X)

    np.random.seed(3)
    nb_epoch_pretraining = 20
    batch_size_pretraining = 256

    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], 
                                      nb_hidden_layers[1:]), start=1):
        print('Training the layer {0}: input {1} -> output {2}'.format(i, n_in, n_out))
        # create AE and training
        input_data = Input(shape=(n_in, ))
        
        ae = Sequential()
        
        encoder = Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='relu')])
        decoder = Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='relu')])

        ae.add(encoder)
        ae.add(decoder)
        
        ae.compile(loss='mse', optimizer='adam')
        ae.fit(X_tmp, X_tmp, batch_size=batch_size_pretraining,
               nb_epoch=nb_epoch_pretraining, verbose=False, shuffle=True)

        ae.summary()
        
        # store trained weight and update training data
        encoder_weights.append(ae.layers[0].get_weights())
        decoder_weights.append(ae.layers[1].get_weights())

        X_tmp = encoder.predict(X_tmp)
    ### END - for
    
    return encoder_weights, decoder_weights
### END - def pretrain


def finetune(X, y, encoder_weights, decoder_weights, group, kfold=True):
    n_in = X.shape[1]
    input_data = Input((n_in, ))
    nb_hidden_layers = [X.shape[1], 500, 100, 20, 2]
    
    ae = Sequential()
    encoder = Sequential()
    decoder = Sequential()
    
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
    
    """
    #encoder.add(Dropout(0.2, input_shape=(X.shape[1],)))
    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:])):
        encoder.add(Dense(output_dim=n_out, input_dim=n_in, 
                          activation='relu', weights=encoder_weights[i]))
   
    for i, (n_out, n_in) in enumerate(zip(nb_hidden_layers[:-1][::-1], nb_hidden_layers[1:][::-1])):
        decoder.add(Dense(output_dim=n_out, input_dim=n_in, activation='relu', 
                          weights=decoder_weights[len(nb_hidden_layers) - i - 2]))
    
    ae.add(encoder)
    ae.add(decoder)
    """
    
    ae.compile(loss='mse', optimizer='rmsprop')
    loss = []
    val_loss = []
    
    # train using stratified kfold
    if kfold == True:
        skf = StratifiedKFold(y, n_folds = 5) # FIXME
        round = 1
        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            history = ae.fit(X_train, X_train, 
                             nb_epoch=80, batch_size=256, 
                             shuffle=True, verbose=False, validation_data=(X_test, X_test))
            loss += history.history['loss']
            val_loss += history.history['val_loss']
            
            print "...finished round {0} of kfold".format(round)
            round += 1
    # train using all data
    
    else:
        # CHANGED: try validation_split param
        ae.fit(X, X, nb_epoch=200, batch_size=256, shuffle=True)
        loss = history.history['loss']

    print "...finished training"
    X_2d = encoder.predict(X)
    X_pred = ae.predict(X)
    
    print "encoded result shape", X_2d.shape

    # plot model loss
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss of ' + group)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('model_loss_' + group + '.png')
    
    return X_2d, mse(X, X_pred)
### END - finetune


def tsne(X):
    tsne = TSNE(learning_rate=1000, n_components=2, verbose=2)
    tsne_result = tsne.fit_transform(X)
    
    return tsne_result
### END - tsne


def mse(true, pred):
    return np.mean(np.mean(np.square(true - pred), axis=-1))


def perform(rank_num, group):
    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
    
    X, y, names = load_data(tetraFile='db2_tetra_{0}_{1}.npy'.format(ranks[rank_num], group), 
                            taxonFile='db2_taxon_{0}_{1}.npy'.format(ranks[rank_num], group), 
                            rank=rank_num+1, minCnt=30)
    
    nb_hidden_layers = [X.shape[1], 500, 100, 20, 2]
    encoder_weights, decoder_weights = pretrain(X)
    kfold = True
    X_2d, score = finetune(X, y, encoder_weights, decoder_weights, group, kfold=kfold)
    
    # plot 2d
    title = "Encoded TNA of " + group
    figName = 'encoder_scaled_2_256-500-100-20-10-2_' + group
    plot2d(X_2d, y, names, title, figName)

    # clustering
    print '...clustering'
    true_clusters = np.unique(y).shape[0] # for cases where there are no 'others'
    d_homo, d_comp, d_vmes, d_ari = dbscan(X_2d, true_clusters, y, title = group + '_ae_scaled')
    a_list = agglomerative(X_2d, true_clusters, y, title = group + '_ae_scaled', connect=False)


    # record to file
    with open('result_score.txt', 'a') as f:
        general_str = "{rank}\t{group}\t{method}\t{size}\t{clusters}\t{mse}\t{layers}\t{kfold}\t".format(rank=ranks[rank_num], group=group, 
                                                                                                         method='ae_scaled', size=X_2d.shape[0],
                                                                                                         clusters=true_clusters, mse=score,
                                                                                                         layers=nb_hidden_layers, kfold=kfold)
        d_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=d_homo, comp=d_comp, vmes=d_vmes, ari=d_ari)
        line = general_str + d_values_str
        for a_value in a_list:
            a_homo, a_comp, a_vmes, a_ari = a_value
            a_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=a_homo, comp=a_comp, vmes=a_vmes, ari=a_ari)
            line += a_values_str
        line += '\n'
        f.write(line)
    ### END - with open
### END - perform


def main():
    
    specificFlag = input("1 for specific, 2 for all under: ")
    # input validity check
    while specificFlag not in [1, 2]:
        specificFlag = eval(input("1 for specific, 2 for all under: "))
    
    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    
    if specificFlag == 1:
        X, y, names = load_data(tetraFile='db2_tetra_top.npy', 
                                taxonFile='db2_taxons_top.npy', 
                                rank=1, minCnt=500)

        rank_num = 0
        group = 'All'
        nb_hidden_layers = [X.shape[1], 500, 100, 20, 2]
        encoder_weights, decoder_weights = pretrain(X)
        kfold = True
        X_2d, score = finetune(X, y, encoder_weights, decoder_weights, group, kfold=kfold)

        # plot 2d
        title = "Encoded TNA of " + group
        figName = 'encoder_scaled_256-500-100-20-10-2_' + group
        plot2d(X_2d, y, names, title, figName)

        # clustering
        print '...clustering'
        true_clusters = np.unique(y).shape[0] # for cases where there are no 'others'
        d_homo, d_comp, d_vmes, d_ari = dbscan(X_2d, true_clusters, y, title = group + '_ae_scaled')
        
        a_list = agglomerative(X_2d, true_clusters, y, title = group + '_ae_scaled')


        # record to file
        with open('result_score.txt', 'a') as f:
            general_str = "{rank}\t{group}\t{method}\t{size}\t{clusters}\t{mse}\t{layers}\t{kfold}\t".format(rank=ranks[rank_num], group=group, 
                                                                                                             method='ae_scaled', size=X_2d.shape[0],
                                                                                                             clusters=true_clusters, mse=score,
                                                                                                             layers=nb_hidden_layers, kfold=kfold)
            d_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=d_homo, comp=d_comp, vmes=d_vmes, ari=d_ari)
            line = general_str + d_values_str
            for a_value in a_list:
                a_homo, a_comp, a_vmes, a_ari = a_value
                a_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=a_homo, comp=a_comp, vmes=a_vmes, ari=a_ari)
                line += a_values_str
            line += '\n'
            f.write(line)
            #a_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\n".format(homo=a_homo, comp=a_comp, vmes=a_vmes, ari=a_ari)
            #line = general_str + d_values_str + a_values_str
            #f.write(line)
        ### END - with open
    ### END - specificFlag

    if specificFlag == 2:
        # loop from phylum to family
        for i in range(1, len(ranks)-2):
            names = np.load('db2_{0}_names.npy'.format(ranks[i])) # e.g. ranks[i] == class

            # peform ae
            for group in names:
                print '\n'
                print '******** ' + group + ' ********'
                perform(i, group)
            ### END - for group
        ### END - for i
    ### END - specific Flag
    
### END - main
    

main()
