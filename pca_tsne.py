from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from SampleGenerator import *

#gen = SampleGenerator()
#X, y = gen.generate()
#phylum_names = gen.getLabelNames()
X = np.load('db2_tetra_top.npy')
y = np.load('db2_phylums_top.npy')
phylum_names = np.load('db2_phylumNames.npy')

pca = PCA(n_components=32)
pca_result = pca.fit(X).transform(X)
print pca_result.shape

# split transformed dataset
#pca_result_odd = pca_result[1::2, :]
#pca_result_even = pca_result[0::2, :]
#phylum_index_odd = phylum_index[1::2]
#phylum_index_even = phylum_index[0::2]

tsne = TSNE(learning_rate=1000, n_components=2, verbose=2)
tsne_result = tsne.fit_transform(pca_result)
#tsne2 = TSNE(learning_rate=100, n_components=2, verbose=2)
#tsne_result_even = tsne2.fit_transform(pca_result_even)

# plot on 2 dimensional space
plt.figure()
for c, i, name in zip ('bgryk', list(range(len(phylum_names))), phylum_names):
    plt.scatter(tsne_result[y == i, 0], tsne_result[y == i, 1], c=c, label=name)
    #plt.scatter(tsne_result_even[phylum_index_even == i, 0], tsne_result_even[phylum_index_even == i, 1], c=c, label=name)
plt.title('PCA-tSNE of tetranucleotide')
plt.legend(loc=3, fontsize=10)
plt.savefig('PCA_TSNE.png')

