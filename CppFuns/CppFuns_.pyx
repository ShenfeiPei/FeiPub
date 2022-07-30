cimport numpy as np
import numpy as np
np.import_array()

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "CppFuns.cpp":
    pass

cdef extern from "CppFuns.h" namespace "cf":
    
    cdef cppclass KNN_Graph[T]:
        vector[vector[int]] NN
        vector[vector[T]] Val

    KNN_Graph[double] knn_graph_tfree(vector[vector[int]] &NN, vector[vector[double]] &NND, vector[double] &NND_k, bool expand)

    KNN_Graph[double] symmetry(vector[vector[int]] &NN, vector[vector[double]] &NND, bool expand)


def knn_graph_tfree_py(np.ndarray[int, ndim=2] NN, np.ndarray[double, ndim=2] NND, np.ndarray[double, ndim=1] NND_k, bool expand):
    ret = knn_graph_tfree(NN, NND, NND_k, expand)
    ret_NN = ret.NN
    ret_Val = ret.Val
    return ret_NN, ret_Val

def symmetry_py(NN, NND, expand):
    ret = symmetry(NN, NND, expand)
    ret_NN = ret.NN
    ret_NND = ret.Val
    return ret_NN, ret_NND


