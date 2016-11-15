import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.lda import LDA
import numpy as np

X = np.load("db2_tetra_top.npy")
y = np.load("db2_phylums_top.npy")
phylum_names = np.load("db2_phylumNames.npy")

lda = LDA(n_components=2)
result = lda.fit(X, y).transform(X)

plt.figure()

for c, i, name in zip("bgryk", list(range(4, -1, -1)), phylum_names):
    plt.scatter(result[y==i, 0], result[y==i, 1], c=c, label=name)
plt.legend(loc=3, fontsize=10)
plt.title("LDA of tetranucleotide")

plt.savefig("LDA selected.png")

