from __future__ import print_function
from glob import glob
import threading
import os
import sys

import numpy as np
import cffi

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


class MulticoreTSNE:
    """
    Compute t-SNE embedding using Barnes-Hut optimization and
    multiple cores (if avaialble).

    Parameters mostly correspond to parameters of `sklearn.manifold.TSNE`.

    The following parameters are unused:
    * n_iter_without_progress
    * min_grad_norm
    * metric
    * method
    
    When `cheat_metric` is true squared equclidean distance is used to build VPTree. 
    Usually leads to same quality, yet much faster.
    """
    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12,
                 learning_rate=200,
                 n_iter=1000,
                 n_iter_without_progress=30,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 init='random',
                 verbose=0,
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 n_jobs=1,
                 cheat_metric=True,
                 use_pca=False,
                 pca_dim=50):
        self.n_components = n_components
        self.angle = angle
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = int(np.random.random() * 1000) if random_state is None else random_state
        self.init = init
        self.embedding_ = None
        self.n_iter_ = None
        self.kl_divergence_ = None
        self.verbose = int(verbose)
        self.cheat_metric = cheat_metric
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        assert isinstance(init, np.ndarray) or init == 'random' or init == 'pca', "init must be 'random' or array"
        if isinstance(init, np.ndarray):
            assert init.ndim == 2, "init array must be 2D"
            assert init.shape[1] == n_components, "init array must be of shape (n_instances, n_components)"
            self.init = np.ascontiguousarray(init, float)

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            """void tsne_run_double(double* X, int N, int D, double* Y,
                                    int no_dims, double perplexity, double theta,
                                    int num_threads, int max_iter, int random_state,
                                    bool init_from_Y, int verbose,
                                    double early_exaggeration, double learning_rate,
                                    double *final_error, int distance, int skip_num_points, int skip_iter);""")

        path = os.path.dirname(os.path.realpath(__file__))

        try:
            sofile = (glob(os.path.join(path, 'libtsne*.so')) +
                      glob(os.path.join(path, '*tsne*.dll')))[0]
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open tsne_multicore shared library')

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, _y=None, skip_num_points=0, skip_iter=0, init_for_skip=None):
        if skip_num_points < 0:
            skip_num_points = 0
        if skip_iter < 0:
            skip_iter = 0

        assert X.ndim == 2, 'X should be 2D array.'

        # X may be modified, make a copy
        X = np.array(X, dtype=float, order='C', copy=True)

        N, D = X.shape

        if skip_num_points > N:
            skip_num_points = N
        if skip_iter > self.n_iter:
            skip_iter = self.n_iter

        if skip_num_points > 0:
            assert isinstance(init_for_skip, np.ndarray), 'init_for_skip should be 2D array.'
            assert init_for_skip.shape == (skip_num_points, self.n_components), 'init_for_skip\'s shape should be ' + str((skip_num_points, self.n_components))

        Y = X.copy()
        if self.use_pca:
            X = X - np.mean(X, axis=0)
            cov_x = np.dot(np.transpose(X), X)
            [eig_val, eig_vec] = np.linalg.eig(cov_x)

            # sorting the eigen-values in the descending order
            eig_vec = eig_vec[:, eig_val.argsort()[::-1]]
            initial_dims = self.pca_dim
            if initial_dims > len(eig_vec):
                initial_dims = len(eig_vec)

            # truncating the eigen-vectors matrix to keep the most important vectors
            eig_vec = np.real(eig_vec[:, :initial_dims])
            X = np.dot(X, eig_vec)
        N, D = X.shape
        init_from_Y = isinstance(self.init, np.ndarray)
        if init_from_Y:
            Y = self.init.copy('C')
            assert X.shape[0] == Y.shape[0], "n_instances in init array and X must match"
        elif self.init == 'random':
            Y = np.zeros((N, self.n_components))
        elif self.init == 'pca':
            Y = Y - np.mean(Y, axis=0)
            cov_x = np.dot(np.transpose(Y), Y)
            [eig_val, eig_vec] = np.linalg.eig(cov_x)

            # sorting the eigen-values in the descending order
            eig_vec = eig_vec[:, eig_val.argsort()[::-1]]
            initial_dims = self.n_components
            if initial_dims > len(eig_vec):
                initial_dims = len(eig_vec)

            # truncating the eigen-vectors matrix to keep the most important vectors
            eig_vec = np.real(eig_vec[:, :initial_dims])
            Y = np.dot(Y, eig_vec)
            init_from_Y = True
        if skip_num_points > 0:
            Y[0:skip_num_points,:] = init_for_skip

        cffi_X = self.ffi.cast('double*', X.ctypes.data)
        cffi_Y = self.ffi.cast('double*', Y.ctypes.data)
        final_error = np.array(0, dtype=float)
        cffi_final_error = self.ffi.cast('double*', final_error.ctypes.data)

        t = FuncThread(self.C.tsne_run_double,
                       cffi_X, N, D,
                       cffi_Y, self.n_components,
                       self.perplexity, self.angle, self.n_jobs, self.n_iter, self.random_state,
                       init_from_Y, self.verbose, self.early_exaggeration, self.learning_rate,
                       cffi_final_error, int(self.cheat_metric), skip_num_points, skip_iter)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        self.embedding_ = Y
        self.kl_divergence_ = final_error
        self.n_iter_ = self.n_iter

        return Y

    def get_nearest_neighbor(self):
        pass


class WeightAssign:
    """
    Assign weight of source data to target data
    """
    def __init__(self,
                 assign_neighbor_number=1):
        self.assign_neighbor_number = assign_neighbor_number
        assert assign_neighbor_number > 0, "assign_neighbor_number must be larger than 0"

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            """void assign_weight_to_nearest_neighbors(double* source_X, int source_N, double* target_X, int target_N,
                                int D, int assign_neighbor_number, double* weight);""")

        path = os.path.dirname(os.path.realpath(__file__))

        try:
            sofile = (glob(os.path.join(path, 'libtsne*.so')) +
                      glob(os.path.join(path, '*tsne*.dll')))[0]
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open tsne_multicore shared library')

    def assign(self, source_data, target_data):
        assert isinstance(source_data, np.ndarray), 'source_data should be 2D array.'
        assert isinstance(target_data, np.ndarray), 'target_data should be 2D array.'
        source_number, source_D = source_data.shape
        target_number, target_D = target_data.shape
        assert source_D == target_D, 'source_data and target_data should be same dim.'
        assert source_number > 0, 'source_data should not be null.'
        assert target_number > 0, 'target_data should not be null.'

        weight = np.zeros((target_number, ))

        cffi_source_data = self.ffi.cast('double*', source_data.ctypes.data)
        cffi_target_data = self.ffi.cast('double*', target_data.ctypes.data)
        cffi_weight = self.ffi.cast('double*', weight.ctypes.data)

        t = FuncThread(self.C.assign_weight_to_nearest_neighbors,
                       cffi_source_data, source_number, cffi_target_data,
                       target_number, source_D, self.assign_neighbor_number,
                       cffi_weight)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        return weight
