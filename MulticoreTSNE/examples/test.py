import gzip
import pickle
import numpy as np
import matplotlib
from cycler import cycler
import urllib
import os
import sys
from sklearn.manifold import TSNE

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", help='Number of threads', default=1, type=int)
parser.add_argument("--n_objects", help='How many objects to use from MNIST', default=-1, type=int)
parser.add_argument("--n_components", help='T-SNE dimensionality', default=2, type=int)
args = parser.parse_args()

def get_mnist():

    if not os.path.exists('mnist.pkl.gz'):
        print('downloading MNIST')
        if sys.version_info >= (3, 0):
            urllib.request.urlretrieve(
            'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')
        else:
            urllib.urlretrieve(
                        'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')
        print('downloaded')

    f = gzip.open("mnist.pkl.gz", "rb")
    if sys.version_info >= (3, 0):
        train, val, test = pickle.load(f, encoding='latin1')
    else:
        train, val, test = pickle.load(f)
    f.close()

    # Get all data in one array
    _train = np.asarray(train[0], dtype=np.float64)
    _val = np.asarray(val[0], dtype=np.float64)
    _test = np.asarray(test[0], dtype=np.float64)
    mnist = np.vstack((_train, _val, _test))

    # Also the classes, for labels in the plot later
    classes = np.hstack((train[1], val[1], test[1]))

    return mnist, classes

def plot(Y, classes, name):
    digits = set(classes)
    fig = plt.figure()
    colormap = plt.cm.spectral
    plt.gca().set_prop_cycle(
        cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 10)]))
    ax = fig.add_subplot(111)
    labels = []
    for d in digits:
        idx = classes == d
        if Y.shape[1] == 1:
            ax.scatter(Y[idx], np.random.randn(Y[idx].shape[0]), s=6)
        else:
            ax.scatter(Y[idx, 0], Y[idx, 1], s=6)
        
        labels.append(d)
    ax.legend(labels, numpoints=1, fancybox=True)
    fig.savefig(name)
    if Y.shape[1] > 2:
        print('Warning! Plot shows only first two components!')

def random_select(source_number, target_number):
    import random
    select = [False] * source_number
    count = 0
    while count < target_number:
        index = int(random.random() * (source_number - 1))
        if select[index]:
            continue
        count += 1
        select[index] = True
    return select

################################################################
def pre_main():
    mnist, classes = get_mnist()
    np.savez('mnist', X = mnist, Y = classes)
    exit(1)
    n_objects = 5000
    target_number = 2000

    mnist = mnist[:n_objects]
    classes = classes[:n_objects]

    # if args.n_objects != -1:
    #     mnist = mnist[:args.n_objects]
    #     classes = classes[:args.n_objects]

    # tsne = TSNE(n_jobs=int(args.n_jobs), verbose=1, n_components=args.n_components, random_state=660)
    selection = random_select(n_objects, target_number)
    sample_X = mnist[selection]
    sample_label = classes[selection]
    total_X = np.zeros(mnist.shape)
    total_X[:target_number, :] = mnist[selection]
    total_X[target_number:, :] = mnist[[not x for x in selection]]
    total_label = np.zeros(classes.shape)
    total_label[:target_number] = classes[selection]
    total_label[target_number:] = classes[[not x for x in selection]]
    
    incremental_tsne = incre_tsne(n_jobs=16, verbose=1, n_components=2, random_state=660, init='pca', cheat_metric=False)
    sklearn_tsne = TSNE(verbose=1, n_components=2, random_state=660, init='pca')
    
    sample_Y = incremental_tsne.fit_transform(sample_X)
    sklearn_sample_Y = sklearn_tsne.fit_transform(sample_X)

    plot(sample_Y, sample_label, 'sample.png')
    plot(sklearn_sample_Y, sample_label, 'sklearn-sample.png')
    
    # np.savetxt('sample_Y.txt', sample_Y)
    total_Y = incremental_tsne.fit_transform(total_X, old_num_points=target_number, join_times=2, init_for_old=sample_Y)
    sklearn_init = np.random.random((n_objects, 2))
    sklearn_init[:target_number] = sample_Y
    sklearn_total_Y = sklearn_tsne.fit_transform(total_X)

    plot(total_Y, total_label, 'total.png')
    plot(sklearn_total_Y, total_label, 'sklearn-total.png')

    # filename = 'mnist_tsne_n_comp=%d.png' % args.n_components
    # np.savetxt('pos.txt', mnist_tsne)
    # plot(mnist_tsne, classes, filename)
    # print('Plot saved to %s' % filename)



from MulticoreTSNE import IncrementalMulticoreTSNE as incre_tsne
pre_main()