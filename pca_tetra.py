from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from clustering import *
from divide_db_level import divide
from supplementary import *

# global variable
global ranks = ["all", "domain", "phylum", "class", "order", "family", "genus", "species"]

def perform(rank_num, group, minCnt=30, tsne=False):
    # perform PCA for group in ranks[rank_num]
    # e.g. perform PCA for Proteobacteria in rank phylum
    print "\n******* {0} *******".format(group)
    
    X, y, names = load_data(tetraFile='db2_tetra_{0}_{1}.npy'.format(ranks[rank_num], group),
                            taxonFile='db2_taxon_{0}_{1}.npy'.format(ranks[rank_num], group),
                            rank=rank_num + 1, minCnt=minCnt)
    # PCA-tSNE
    if tsne:
        method = "pca_tsne_scaled"
        pca = PCA(n_components=32)
        X_32d = pca.fit_transform(X)
        pca_tsne = TSNE(learning_rate=800, n_components=2, verbose=1)
        X_2d = pca_tsne.fit_transform(X_32d)
        title = "PCA-tSNE of TNA of " + group
        figName = "PCA-tSNE_{0}_{1}".format(ranks[rank_num][0], group)
    ### END - if tsne

    # only PCA
    else:
        method = "pca_scaled"
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        title = "PCA of TNA of " + group
        figName = 'PCA_{0}_{1}'.format(ranks[rank_num][0], group)

        # remap to original space
        X_pred = pca.inverse_transform(X_2d)
    ### END - else

    plot2d(X_2d, y, names, title, figName)

    # clustering
    print '...clustering'
    name = '{0}_{1}'.format(ranks[rank_num][0], group)
    clust = Clustering(name, method, X_2d, y)
    d_homo, d_comp, d_vmes, d_ari = clust.dbscan(10)
    eval_tuple = clust.agglomerative(linkage='all', connect=False)
    clust.plotCluster(alg='all')

    # record
    with open('result_score.txt', 'a') as f:
        if tsne:
            mse_result = '-'
        else:
            mse_result = mse(X, X_pred)

        general_str = "{rank}\t{group}\t{method}\t{size}\t\t{mse}\t\t\t".format(rank=ranks[rank_num],
                                                                                group=group,
                                                                          method=method,
                                                                          size=X_2d.shape[0], mse=mse_result)
        d_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=d_homo, comp=d_comp,
                                                                                    vmes=d_vmes, ari=d_ari)
        line = general_str + d_values_str
        
        for i in range(3):
            a_homo, a_comp, a_vmes, a_ari = eval_tuple[i], eval_tuple[i+1], eval_tuple[i+2], eval_tuple[i+3]
            a_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=a_homo, comp=a_comp,
                                                                                        vmes=a_vmes, ari=a_ari)
            line += a_values_str
        ### END - for i
        line += '\n'
        f.write(line)
    ### END - with open
### END - perform


def main():
    flag = input("1 for specific, 2 for all under 3 for etc: ")
    
    if flag == 2:
        # loop from phylum to family
        for i in range(2, len(ranks) - 2):

            # generate taxon and tetra file for rank under this rank
            # e.g. if current rank is class, then generate datasets for order
            names = np.load('db2_{0}_names.npy'.format(ranks[i]))  # e.g. ranks[i] == class

            # peform pca
            for group in names:
                print
                '\n'
                print
                '******** ' + group + ' ********'
                perform(i, group)
                ### END - for group
                ### END - for i
    ### END - if flag == 2

    elif flag == 1:  # specific
        rankName = input("taxonomical rank: ")
        taxonName = input("taxon name: ")
        perform(ranks.index(rankName), taxonName, minCnt=30)
    ### END - elif flag == 1

    elif flag == 3:
        genus_list = np.load('db2_genus_list.npy')

        rank_num = 6  # genus
        for group in genus_list:
            perform(rank_num, group, minCnt=10, tsne=False)
            perform(rank_num, group, minCnt=10, tsne=True)
        ### END - for group
    ### END - elif flag == 3
### END - main

main()
