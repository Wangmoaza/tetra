from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import numpy as np
from clustering import *
from supplementary import *
from encoder_pretrain import *

species = "s_Vibrio_cholerae"

def perform_PCA(X, dim=2, tsne=False):
    # PCA-tSNE
    if tsne:
        method = "pca_tsne_scaled"
        pca = PCA(n_components=32)
        X_32d = pca.fit_transform(X)
        pca_tsne = TSNE(learning_rate=800, n_components=dim, verbose=1)
        X_2d = pca_tsne.fit_transform(X_32d)
    ### END - if tsne

    # only PCA
    else:
        method = "pca_scaled"
        pca = PCA(n_components=dim)
        X_2d = pca.fit_transform(X)
    ### END - else

    # clustering
    print('***** ' + method + ' *****')
    cluster(X_2d, method)
    np.save("{0}_{1}_X_2d".format(species, method), X_2d)
### END - def perform_PCA


def perform_AE(X, dim=2, tsne=False):
    y = np.zeros(shape=X.shape[0], dtype=int)
    
    if tsne:
        hidden_layers = [X.shape[1], 500, 100, 32]
        encoder_weights, decoder_weights = pretrain(X, hidden_layers)
        X_32d = ae(X, encoder_weights, decoder_weights, hidden_layers)

        ae_tsne = TSNE(n_components=dim, learning_rate=800, verbose=1)
        X_2d = ae_tsne.fit_transform(X_32d)

        method = 'ae_tsne_scaled'
    ### END - if tsne

    else:
        hidden_layers = [X.shape[1], 500, 100, 20, dim]
        encoder_weights, decoder_weights = pretrain(X, hidden_layers)
        X_2d = ae(X, encoder_weights, decoder_weights, hidden_layers)
        
        method = 'ae_scaled'
    ### END - else

    print('***** ' + method + ' *****')
    cluster(X_2d, method)
    np.save("{0}_{1}_X_2d".format(species, method), X_2d)
### END - def perform_AE


def cluster(X_2d, method):
    for i in range(2, 6):
        print("n_clusters\t{0}".format(i))
        clust = Clustering(species, method, X_2d, None, n_clusters=i)

        if i == 2:
            eval_tuple_db = clust.dbscan(5)

        clust.dbscan(5)
        eval_tuple_agg = clust.agglomerative(linkage='all', connect=True)
        string = ""
        for j in range(len(eval_tuple_agg)):
            string += '{0}\t'.format(eval_tuple_agg[j])

        clust.plotCluster(alg='all', num=i)
        clust.plotCluster(alg='ward', num=i)
        
        with open("h_pylori_record.txt", "a") as f:
            f.write("{0}\t{1}\t{2}\t{3}\n".format(method, i, eval_tuple_db[0], string[:-1]))
### END - def cluster

def ae(X, encoder_weights, decoder_weights, hidden_layers):
    # finetune
    n_in = X.shape[1]
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
    # train using all data
    ae.fit(X, X, nb_epoch=200, batch_size=32, shuffle=True, verbose=0)
    X_pred = encoder.predict(X)
    return X_pred
### END - def ae


def main():
    X = np.load('Vibrio_cholerae_tetra.npy')
    perform_PCA(X, tsne=False)
    #perform_PCA(X, tsne=True)
    #perform_AE(X, tsne=False)
    #perform_AE(X, tsne=True)

    
if __name__ == '__main__':
    main()