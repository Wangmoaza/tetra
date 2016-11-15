from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tetra_freq = np.load('tetrafreq.npy')
phylum_index = np.load('phylumIndex.npy')
phylum_names = np.load('phylumNames.npy')

lle = LocallyLinearEmbedding(n_components=2)
lle_result = lle.fit_transform(tetra_freq)

plt.figure()
for c, i, name in zip ("bgrcmykw", list(range(7, -1, -1)), phylum_names):
    plt.scatter(lle_result[phylum_index == i, 0], lle_result[phylum_index == i, 1], c=c, label=name)
plt.title('LLE of tetranucleotide')
plt.legend(loc=3, fontsize=10)
plt.savefig('LLE.png')


