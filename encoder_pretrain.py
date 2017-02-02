from keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input
from keras.utils import np_utils
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.manifold import TSNE
from keras import regularizers
from keras.callbacks import TensorBoard
from time import time
from clustering import *
from supplementary import *

ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]

def pretrain(X, hidden_layers):
    encoder_weights = []
    decoder_weights = []
    X_tmp = np.copy(X)

    np.random.seed(3)
    nb_epoch_pretraining = 20
    batch_size_pretraining = 256

    for i, (n_in, n_out) in enumerate(zip(hidden_layers[:-1],
                                          hidden_layers[1:]), start=1):
        print('Training the layer {0}: input {1} -> output {2}'.format(i, n_in, n_out))
        # create AE and training
        input_data = Input(shape=(n_in,))

        ae = Sequential()

        encoder = Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='relu')])
        decoder = Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='relu')])

        ae.add(encoder)
        ae.add(decoder)

        ae.compile(loss='mse', optimizer='adam')
        ae.fit(X_tmp, X_tmp, batch_size=batch_size_pretraining,
               nb_epoch=nb_epoch_pretraining, verbose=False, shuffle=True)

        #ae.summary()

        # store trained weight and update training data
        encoder_weights.append(ae.layers[0].get_weights())
        decoder_weights.append(ae.layers[1].get_weights())

        X_tmp = encoder.predict(X_tmp)
    ### END - for

    return encoder_weights, decoder_weights
### END - def pretrain


def finetune(X, y, encoder_weights, decoder_weights, group, hidden_layers, kfold=True):
    n_in = X.shape[1]

    """
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
    ae = Sequential()
    encoder = Sequential()
    decoder = Sequential()

    for i, (n_in, n_out) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
        encoder.add(Dense(output_dim=n_out, input_dim=n_in,
                          activation='relu', weights=encoder_weights[i]))

    for i, (n_out, n_in) in enumerate(zip(hidden_layers[:-1][::-1], hidden_layers[1:][::-1])):
        decoder.add(Dense(output_dim=n_out, input_dim=n_in, activation='relu',
                          weights=decoder_weights[len(hidden_layers) - i - 2]))

    ae.add(encoder)
    ae.add(decoder)

    ae.compile(loss='mse', optimizer='rmsprop')
    loss = []
    val_loss = []

    # train using stratified kfold
    if kfold:
        skf = StratifiedKFold(y, n_folds=5)  # FIXME
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
        # END - for
    ### END - if

    # train using all data
    else:
        # CHANGED: try validation_split param
        ae.fit(X, X, nb_epoch=200, batch_size=256, shuffle=True)
        loss = history.history['loss']
    ### END - else

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
### END - def finetune


def perform(rank_num, group, minCnt=30, dim=2, tsne=False):
    print "\n\n******* {0} *******".format(group)
    
    X, y, names = load_data(tetraFile='db2_tetra_{0}_{1}.npy'.format(ranks[rank_num], group),
                            taxonFile='db2_taxon_{0}_{1}.npy'.format(ranks[rank_num], group),
                            rank=rank_num + 1, minCnt=minCnt)

    kfold = True
    
    # validity check
    if X.shape[0] == 0:
        print "Error: sample size is 0"
        return
    
    if tsne:
        hidden_layers = [X.shape[1], 500, 100, 32]
        encoder_weights, decoder_weights = pretrain(X, hidden_layers)
        X_32d, score = finetune(X, y, encoder_weights, decoder_weights, group, hidden_layers, kfold=kfold)
        ae_tsne = TSNE(n_components=dim, learning_rate=800, verbose=1)
        X_2d = ae_tsne.fit_transform(X_32d)

        title = "AE-TSNE TNA of {0}_{1}".format(ranks[rank_num][0], group)
        figName = "ae_tsne_scaled_{0}_{1}".format(ranks[rank_num][0], group)
        method = 'ae_tsne_scaled'
        if dim > 2:
            figName += '_3d'
        
    ### END - if tsne

    else:
        hidden_layers = [X.shape[1], 500, 100, 20, dim]
        encoder_weights, decoder_weights = pretrain(X, hidden_layers)
        X_2d, score = finetune(X, y, encoder_weights, decoder_weights, group, hidden_layers, kfold=kfold)

        title = "AE TNA of {0}_{1}".format(ranks[rank_num][0], group)
        figName = "ae_scaled_{0}_{1}".format(ranks[rank_num][0], group)
        method = 'ae_scaled'
    ### END - else

    
    #FIXME
    ##### Early stopping for runtime check
    return
    ######

    # plot
    if dim == 2:
        plot2d(X_2d, y, names, title, figName)
    elif dim == 3:
        plot3d(X_2d, y, title, figName, names=names)

    # clustering
    print '...clustering'
    name = '{0}_{1}'.format(ranks[rank_num][0], group)
    clust = Clustering(name, method, X_2d, y)
    d_ami, d_nmi, d_vmes, d_ari = clust.dbscan(5)
    eval_tuple = clust.agglomerative(linkage='ward', connect=True)
    clust.plotCluster(alg='ward')

    # record to file
    with open('result_score.txt', 'a') as f:
        if tsne:
            mse_result = '-'
        else:
            mse_result = score

        general_str = "{rank}\t{group}\t{method}\t{size}\t{clusters}\t{mse}\t{dim}\t{layers}\t{kfold}\t".format(
            rank=ranks[rank_num], group=group,
            method=method, size=X_2d.shape[0],
            clusters=np.unique(y).shape[0], 
            mse=mse_result, dim=dim,
            layers=hidden_layers, kfold=kfold)
        d_values_str = "{ami:0.3f}\t{nmi:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(ami=d_ami, nmi=d_nmi,
                                                                                    vmes=d_vmes, ari=d_ari)
        line = general_str + d_values_str

        for i in range(len(eval_tuple) / 4):
            a_ami, a_nmi, a_vmes, a_ari = eval_tuple[i], eval_tuple[i+1], eval_tuple[i+2], eval_tuple[i+3]
            a_values_str = "{ami:0.3f}\t{nmi:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(ami=a_ami, nmi=a_nmi,
                                                                                        vmes=a_vmes, ari=a_ari)
            line += a_values_str
        ### END - for i
        line += '\n'
        f.write(line)
        ### END - with open
    ### END - with open
### END - perform


def main():
    specificFlag = input("1 for specific, 2 for all, under 3 for etc.: ")
    # input validity check
    while specificFlag not in [1, 2, 3]:
        specificFlag = eval(input("1 for specific, 2 for all under, 3 for etc.: "))

    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]

    
    if specificFlag == 1:
        rankName = input("taxonomical rank: ")
        taxonName = input("taxon name: ")
        perform(ranks.index(rankName), taxonName, minCnt=30)
    ### END - specificFlag == 1

    elif specificFlag == 2:
        # loop from phylum to family
        for i in range(1, len(ranks) - 2):
            names = np.load('db2_{0}_names.npy'.format(ranks[i]))  # e.g. ranks[i] == class

            # peform ae
            for group in names:
                print '\n'
                print '******** ' + group + ' ********'
                perform(i, group)
            ### END - for group
        ### END - for i
    ### END - specific Flag == 2

    elif specificFlag == 3:
        genus_list = np.load('db2_genus_list.npy')
        rank_num = ranks.index('genus')
        
        t0 = time()
        for group in genus_list:
            #perform(rank_num, group, minCnt=10, dim=2, tsne=False)
            perform(rank_num, group, minCnt=10, dim=2, tsne=True)
        ### END - for group
    ### END - specificFlag == 3
    print "running time: {0}s".format(time() - t0)

### END - main

if __name__ == '__main__':
    main()
