import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from .CppFuns.CppFuns_ import symmetry_py
from .CppFuns.CppFuns_ import knn_graph_tfree_py
from . import funs as Funs

def get_anchor(X, m, way="random"):
    """
    X: n x d,
    m: the number of anchor
    way: [k-means, k-means2, k-means++, k-means++2, random]
    """
    if way == "k-means":
        A = KMeans(m, init='random').fit(X).cluster_centers_
    elif way == "k-means2":
        A = KMeans(m, init='random').fit(X).cluster_centers_
        D = EuDist2(A, X)
        ind = np.argmin(D, axis=1)
        A = X[ind, :]
    elif way == "k-means++":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
    elif way == "k-means++2":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
        D = EuDist2(A, X)
        A = np.argmin(D, axis=1)
    elif way == "random":
        ids = random.sample(range(X.shape[0]), m)
        A = X[ids, :]
    else:
        raise SystemExit('no such options in "get_anchor"')
    return A


def knn_f(X, knn, squared=True, self_include=True):
    t_start = time.time()

    D_full = EuDist2(X, X, squared=squared)
    np.fill_diagonal(D_full, -1)
    NN_full = np.argsort(D_full, axis=1)
    np.fill_diagonal(D_full, 0)

    if self_include:
        NN = NN_full[:, :knn]
    else:
        NN = NN_full[:, 1:(knn+1)]

    NND = Funs.matrix_index_take(D_full, NN)

    NN = NN.astype(np.int32)
    NND = NND.astype(np.float64)

    t_end = time.time()
    t = t_end - t_start

    return NN, NND, t


def knn_graph_gaussian(X, knn, t_way="mean", self_include=False, isSym=True):
    """
    :param X: data matrix of n by d
    :param knn: the number of nearest neighbors
    :param t_way: the bandwidth parameter
    :param self_include: weather xi is among the knn of xi
    :param isSym: True or False, isSym = True by default
    :return: A, a matrix (graph) of n by n
    """
    N = X.shape[0]
    NN, NND, time1 = knn_f(X, knn, squared=True, self_include=self_include)

    Val = dist2sim_kernel(NND, t_way=t_way)

    A = np.zeros((N, N))
    Funs.matrix_index_assign(A, NN, Val)
    np.fill_diagonal(A, 0)

    if isSym:
        A = (A + A.T) / 2

    return A

def dist2sim_kernel(NND, t_way="mean"):
    if t_way == "mean":
        t = np.mean(NND)
    elif t_way == "median":
        t = np.median(NND)
    else:
        raise SystemExit('no such options in "dist2sim_kernel, t_way"')

    Val = np.exp(-NND / (2 * t ** 2))

    return Val

def knn_graph_tfree(X, knn, self_include=False, isSym=True):
    """
    :param X: data matrix of n by d
    :param knn: the number of nearest neighbors
    :param self_include: weather xi is among the knn of xi
    :param isSym: True or False, isSym = True by default
    :return: A, a matrix (graph) of n by n
    """
    t_start = time.time()

    N = X.shape[0]
    NN_K, NND_K, time1 = knn_f(X, knn + 1, squared=True, self_include=self_include)

    NN = NN_K[:, :knn]
    NND = NND_K[:, :knn]
    NND_k = NND_K[:, knn]

    Val = dist2sim_t_free(NND, NND_k)

    A = np.zeros((N, N))
    Funs.matrix_index_assign(A, NN, Val)
    np.fill_diagonal(A, 0)

    if isSym:
        A = (A + A.T) / 2
    
    t_end = time.time()
    t = t_end - t_start
    return A, t

def dist2sim_t_free(NND, NND_k):
    knn = NND.shape[1]

    Val = NND_k.reshape(-1, 1) - NND

    Val[Val[:, 0] < 1e-6, :] = 1.0/knn

    Val = Val / (np.sum(Val, axis=1).reshape(-1, 1))
    return Val

def kng_anchor(X, Anchor: np.ndarray, knn=20, way="gaussian", t_way="mean", shape=None):
    """
    :param X: data matrix of n by d
    :param Anchor: Anchor set, m by d
    :param knn: the number of nearest neighbors
    :param way: one of ["gaussian", "t_free"]
        "t_free" denote the method proposed in : "The constrained laplacian rank algorithm for graph-based clustering"
        "gaussian" denote the heat kernel
    :param t_way: only needed by gaussian, the bandwidth parameter
    :return: A, a matrix (graph) of n by m
    """
    N = X.shape[0]
    anchor_num = Anchor.shape[0]

    # NN_K, NND_K
    D = EuDist2(X, Anchor, squared=True)  # n x m
    NN_full = np.argsort(D, axis=1)
    NN_K = NN_full[:, :(knn+1)]  # xi isn't among neighbors of xi
    NND_K = Funs.matrix_index_take(D, NN_K)

    # NN, NND, NND_k
    NN = NN_K[:, :knn]
    NND = NND_K[:, :knn]
    NND_k = NND_K[:, knn]

    if way=="gaussian":
        Val = dist2sim_kernel(NND, t_way=t_way)
    elif way=="t_free":
        Val = dist2sim_t_free(NND, NND_k)
    else:
        raise SystemExit('no such options in "get_anchor"')

    A = np.zeros((N, anchor_num))
    Funs.matrix_index_assign(A, NN, Val)
    return A
