import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import mpl_toolkits.mplot3d.axes3d as p3
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# load dataset
def load_data(tetraFile, taxonFile, rank, minCnt=10):
    X = np.load(tetraFile)

    # standardize data
    X_scaled = preprocessing.scale(X) + 1

    taxons = np.load(taxonFile)
    names = getTopNames(taxons, rank, minCnt)
    y = label2num(taxons, names, rank)

    # exclude Others
    condition = y != len(names) - 1
    X_nonOther = np.ndarray(shape=(0, X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        if condition[i]:
            X_nonOther = np.vstack((X_nonOther, X_scaled[i,]))

    y_nonOther = np.extract(condition, y)
    print X_nonOther.shape, y_nonOther.shape
    return X_nonOther, y_nonOther, names[:-1]
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
            print(label, dic.get(label))
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
            indexList.append(len(names) - 1)  # last element is 'Others'
    ### END - for i

    for i in range(len(names)):
        print i, indexList.count(i)

    return np.asarray(indexList)
### END - def label2num


def plot2d(result, y, names, title, figName):
    fig = plt.figure(figsize=(400, 400))
    ax = plt.subplot(111)
    for i, name in enumerate(names):
        ax.scatter(result[y == i, 0], result[y == i, 1], 
                    color=plt.cm.jet(np.float(i) / len(names)), 
                    label=name)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'x-small')
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.savefig(figName + '.png')
### END - def plot2d


def plot3d(X, y, title, figName, names=None, savefig=True):
    # validity check
    if X.shape[1] != 3:
        print "Error: data not 3D"
        return
    
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
    markers = markers * 5
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if names == None:
        names = np.unique(y)
        for i, name in enumerate(names):
            ax.scatter(X[y==i, 0], X[y==i, 1], X[y==i, 2], 
                       color=plt.cm.jet(np.float(i) / len(names)), 
                       marker=markers[i])
    ### END - if
    else:
        for i, name in enumerate(names):
            ax.scatter(X[y==i, 0], X[y==i, 1], X[y==i, 2], 
                       color=plt.cm.jet(np.float(i) / len(names)), 
                       marker=markers[i])
    ### END - else
    if savefig:
        plt.title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.savefig('{0}_3d.png'.format(figName))
        return None
    
    else:
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        return ax
### END - def plot3d

def plot3d_ani(X, y, title, figName, names=None):
        ax = plot3d(X, y, title, figName, names=names, savefig=False)
        angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
        # create an animated gif (20ms between frames)
        rotanimate(ax, angles, figName + '.gif',delay=20) 
        
##### code from https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/
def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = '%s%03d.png' % (prefix, i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files

##### code from https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/
def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              % (delay, loop, " ".join(files), output))


##### MAIN FUNCTION
##### code from https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/
def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.gif': make_gif}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)

def mse(true, pred):
    return np.mean(np.mean(np.square(true - pred), axis=-1))