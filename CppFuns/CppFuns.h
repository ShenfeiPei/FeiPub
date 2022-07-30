#ifndef CPPFUNS_H_
#define CPPFUNS_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>


using namespace std;

namespace cf{

    template<typename T>
    struct KNN_Graph{
        vector<vector<int>> NN;
        vector<vector<T>> Val;
        // vector<vector<double>> A;
    };

    chrono::time_point<chrono::steady_clock> time_now();
    double time_diff(chrono::time_point<chrono::steady_clock> t1, chrono::time_point<chrono::steady_clock> t2);

    template <typename T>
    void symmetry_sub(vector<unordered_map<int, T>> &G, vector<unordered_map<int, T>> &RG, int i, 
                      vector<int> &nn, vector<T> &nnd, bool expand);

    template <typename T>
    KNN_Graph<T> symmetry(vector<vector<int>> &NN, vector<vector<T>> &NND, bool expand);

    void check_NN(vector<vector<int>> &NN, bool self_include);

    KNN_Graph<double> knn_graph_tfree(vector<vector<int>> &NN, vector<vector<double>> &NND, vector<double> &NND_k, bool expand);

    template<typename T>
    void show_vec(vector<T> &Vec, unsigned int n);

    template<typename T>
    void read_2Dvec(std::string name, unsigned int K, std::vector<std::vector<T>> &M);

    template<typename T>
    void show_2Dvec(std::vector<std::vector<T>> &Vec, unsigned int n, unsigned int m);

    template<typename T>
    void argsort_f(vector<T> &v, vector<int> &ind);

    double maximum_2Dvec(std::vector<std::vector<double>> &Vec);

    template<typename T>
    void write_vec(std::string name, std::vector<T> &y);
}

#endif
