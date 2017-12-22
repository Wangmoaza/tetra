from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import numpy as np
from clustering import *
from supplementary import *
from encoder_pretrain import *

species = "s_Campylobacter_jejuni"

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
    np.save("{0}_{1}_X_2d".format(species, method), X_2d) # save reduced 2d tetra file
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
        
        with open("c_jejuni_record.txt", "a") as f:
            f.write("{0}\t{1}\t{2}\t{3}\n".format(method, i, eval_tuple_db[0], string[:-1]))
### END - def cluster

def scipy_cluster(X_2d, method="", ids=None):
    clust = Scipy_Clustering(X_2d, name=species, method=method, ids=ids, n_clusters=2)
    clust.cluster()
    #clust.plot_cluster()
    #clust.plot_dendrogram(truncate_mode='lastp', max_d=250)
    #for i in range(3, 5):
    #    clust.set_n_clusters(i)
    #    clust.plot_cluster()
    with open("{0}_{1}_newick.nwk".format(species, method), "w") as f:
        f.write(clust.dendrogram_to_newick())
### END - def scipy_cluster

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

def screen_non_pylori():
    true_taxon_sorted = np.load('h_pylori_true_taxon_sorted.npy')
    file_list = ['s_Helicobacter_pylori_ae_scaled_X_2d.npy', 
                 's_Helicobacter_pylori_ae_tsne_scaled_X_2d.npy',
                 's_Helicobacter_pylori_pca_scaled_X_2d.npy',
                 's_Helicobacter_pylori_pca_tsne_scaled_X_2d.npy']
    method_list = ['ae_scaled_screen', 'ae_tsne_scaled_screen', 'pca_scaled_screen', 'pca_tsne_scaled_screen']
    
    # taxon to label index
    names = np.unique(true_taxon_sorted)
    print names
    names = names.tolist()
    y = []
    for item in true_taxon_sorted:
        y.append(names.index(item))
    y = np.asarray(y)
    print "y shape: ", y.shape
    
    ids = np.load('Helicobacter_pylori_ids.npy')
    idx = np.argsort(ids)
    for i in range(4):
        X_2d = np.load(file_list[i])
        X_2d_sorted = X_2d[idx]
        method = method_list[i]
        figName = "s_Helicobacter_pylori_" + method
        plot2d(X_2d_sorted, y, names, figName, figName)
    ### END - for i
### END - def screen

def compare(method1, method2, fig=False):
    X1 = np.load('{0}_{1}_X_2d.npy'.format(species, method1))
    X2 = np.load('{0}_{1}_X_2d.npy'.format(species, method2))
    
    print 'n_cluster\tHomo\tCompl\tNMI\tARI'
    for i in range(2, 6):
        clust1 = Clustering(species, method1, X1, None, n_clusters=i)
        clust2 = Clustering(species, method2, X2, None, n_clusters=i)
        
        clust1.agglomerative(linkage='ward')
        clust2.agglomerative(linkage='ward')
        
        label1 = clust1.pred_labels('ward')
        label2 = clust2.pred_labels('ward')
        
        
        if i == 3 and fig:
            names = np.unique(label1)
            figName = '{0}_{1}_on_{2}'.format(species, method1, method2)
            plot2d(X2, label1, names, figName, figName)

            names = np.unique(label2)
            figName = '{0}_{1}_on_{2}'.format(species, method2, method1)
            plot2d(X1, label2, names, figName, figName)
    
        print '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(i, metrics.homogeneity_score(label1, label2),
                                                metrics.completeness_score(label1, label2),
                                                metrics.normalized_mutual_info_score(label1, label2),
                                                metrics.adjusted_rand_score(label1, label2))
    ### END - for i
    

### END - def compare

def main():
    #X = np.load('Campylobacter_jejuni_tetra.npy')
    #perform_PCA(X, tsne=False)
    #perform_PCA(X, tsne=True)
    #perform_AE(X, tsne=False)
    #perform_AE(X, tsne=True)
    #screen_non_pylori()
    #compare('ae_tsne_scaled', 'pca_tsne_scaled', fig=True)
    #compare('ae_tsne_scaled', 'pca_tsne_scaled')

    ids = np.load('s_Campylobacter_jejuni_profile.npy')
    method_list = ['ae_tsne_scaled', 'ae_scaled', 'pca_scaled', 'pca_tsne_scaled']
    
    for method in method_list:
        X_2d = np.load('s_Campylobacter_jejuni_{0}_X_2d.npy'.format(method))
        scipy_cluster(X_2d, method=method, ids=ids)

    
if __name__ == '__main__':
    main()
