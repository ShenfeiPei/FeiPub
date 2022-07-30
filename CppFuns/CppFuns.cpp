#include "CppFuns.h"

namespace cf{

    chrono::time_point<chrono::steady_clock> time_now(){
        chrono::time_point<chrono::steady_clock> t1 = chrono::steady_clock::now();
        return t1;
    }
    double time_diff(chrono::time_point<chrono::steady_clock> t1, chrono::time_point<chrono::steady_clock> t2){
        double ret = chrono::duration<double>(t2 - t1).count();
        return ret;
    }

    template <typename T>
    void symmetry_sub(vector<unordered_map<int, T>> &G, vector<unordered_map<int, T>> &RG, int i, 
                      vector<int> &nn, vector<T> &nnd, bool expand){

        vector<int> ori_nn = nn;
        vector<T> ori_nnd = nnd;

        nn.clear();
        nnd.clear();

        int nb;
        T d1, d2;
        for (int k = 0; k < ori_nn.size(); k++){
            // i -> nb, dist = d1
            nb = ori_nn[k];
            d1 = ori_nnd[k];

            auto it = G[nb].find(i);
            if (it != G[nb].end()){
                // nb -> i, dist = d2
                d2 = it->second;

                // i-> nb, nb-> i
                nn.push_back(nb);
                nnd.push_back((d1 + d2)/((T)2));

            }else if (expand){
                // i-> nb,   nb !-> i
                nn.push_back(nb);
                nnd.push_back(d1);
            }
        }
        if (expand) for (auto ele: RG[i]){
            // nb -> i, dist = d1
            nb = ele.first;
            d1 = ele.second;

            auto it = G[i].find(nb);
            if (it == G[i].end()){
                // nb -> i, i !-> nb
                nn.push_back(nb);
                nnd.push_back(d1);
            }
        }
    }
    template void symmetry_sub<float>(vector<unordered_map<int, float>> &G, vector<unordered_map<int, float>> &RG, int i, 
                                      vector<int> &nn, vector<float> &nnd, bool expand);
    template void symmetry_sub<double>(vector<unordered_map<int, double>> &G, vector<unordered_map<int, double>> &RG, int i, 
                                      vector<int> &nn, vector<double> &nnd, bool expand);

    template <typename T>
    KNN_Graph<T> symmetry(vector<vector<int>> &NN, vector<vector<T>> &NND, bool expand){
        int nb;

        vector<unordered_map<int, T>> G;
        vector<unordered_map<int, T>> RG;

        G.resize(NN.size());
        for (unsigned int i = 0; i < NN.size(); i++){
            for (unsigned int k = 0; k < NN[i].size(); k++){
                nb = NN[i][k];
                G[i][nb] = NND[i][k];
            }
        }
        if (expand){
            RG.resize(NN.size());
            for (unsigned int i = 0; i < NN.size(); i++){
                for (unsigned int k = 0; k < NN[i].size(); k++){
                    nb = NN[i][k];
                    RG[nb][i] = NND[i][k];
                }
            }
        }

        #pragma omp parallel for
        for (unsigned int i = 0; i < NN.size(); i++){
            symmetry_sub(G, RG, i, NN[i], NND[i], expand);
        }

        KNN_Graph<T> tmp;
        tmp.NN = NN;
        tmp.Val = NND;
        return tmp;
    }
    template KNN_Graph<float> symmetry<float>(vector<vector<int>> &NN, vector<vector<float>> &NND, bool expand);
    template KNN_Graph<double> symmetry<double>(vector<vector<int>> &NN, vector<vector<double>> &NND, bool expand);

    // convert distance to similarity
    void knn_graph_tfree_sub(vector<double> &d, vector<double> &val, double d_max){
        transform(d.begin(), d.end(), val.begin(), [d_max](double di){return d_max - di;});
        double s = accumulate(val.begin(), val.end(), (double) 0);
        if (s > 1e-6){
            transform(val.begin(), val.end(), val.begin(), [s](double vi){return vi/s;});
        }else{
            fill(val.begin(), val.end(), 1.0/val.size());
        }
    }

    void check_NN(vector<vector<int>> &NN, bool self_include){
        for (unsigned int i = 0; i < NN.size(); i++){
            if (self_include && NN[i][0] != i){
                cout << "NN is not eligible" << endl;
                exit(EXIT_FAILURE);
            }
            if (!self_include){
                if (NN[i][0] == i){
                    auto it = NN[i].begin();
                    NN[i].erase(it);
                    // cout << "NN is not eligible" << endl;
                    // exit(EXIT_FAILURE);
                }
            }
        }
    }

    KNN_Graph<double> knn_graph_tfree(vector<vector<int>> &NN, vector<vector<double>> &NND, vector<double> &NND_k, bool expand){

        check_NN(NN, false);

        vector<vector<double>> Val = NND;
        #pragma omp parallel for
        for (unsigned int i = 0; i < NND.size(); i++){
            knn_graph_tfree_sub(NND[i], Val[i], NND_k[i]);
        }
        KNN_Graph<double> G = symmetry(NN, Val, expand);
        return G;
    }

    template<typename T>
    void read_2Dvec(std::string name, unsigned int K, std::vector<std::vector<T>> &M){
        unsigned int N = M.size();
        unsigned int k = M[0].size();

        std::fstream in_f;
        T *tmp_K = new T[K];
        in_f.open(name.c_str(), std::ios::in | std::ios::binary);
        if(!in_f.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}

        for (unsigned int i = 0; i < N; i++){
            in_f.read((char*)tmp_K, sizeof(T)*K);
            M[i].assign(tmp_K, tmp_K + k);
        }
        in_f.close();
    }
    template void read_2Dvec<int>   (std::string name, unsigned int K, std::vector<std::vector<int>> &M);
    template void read_2Dvec<double>(std::string name, unsigned int K, std::vector<std::vector<double>> &M);

    template<typename T>
    void show_2Dvec(std::vector<std::vector<T>> &Vec, unsigned int n, unsigned int m){
        unsigned int N = Vec.size();
        unsigned int M = Vec[0].size();
        if (n > N || m > M) std::cout << "n (m) must be less or equal than N (M)." << std::endl;

        for (unsigned int i = 0; i < n; i++){
            for (unsigned int j = 0; j < m; j++){
                std::cout << Vec[i][j] << ", ";
            }
            std::cout << std::endl;
        }
    }
    template void show_2Dvec<int>(std::vector<std::vector<int>> &Vec, unsigned int n, unsigned int m);
    template void show_2Dvec<double>(std::vector<std::vector<double>> &Vec, unsigned int n, unsigned int m);

    template<typename T>
    void show_vec(vector<T> &Vec, unsigned int n){
        unsigned int N = Vec.size();
        if (n > N) cout << "n must be less or equal than N." << endl;

        for (unsigned int i = 0; i < n; i++){
            cout << Vec[i] << ", ";
        }
        cout << endl;
    }
    template void show_vec<int>(vector<int> &Vec, unsigned int n);
    template void show_vec<float>(vector<float> &Vec, unsigned int n);
    template void show_vec<double>(vector<double> &Vec, unsigned int n);



    template<typename T>
    void argsort_f(vector<T> &v, vector<int> &ind){
        std::iota(ind.begin(), ind.end(), 0);
        // for (int i = 0; i < v.size(); i++) ind[i] = i;
        std::sort(ind.begin(), ind.end(), [&v](int i1, int i2){ return v[i1] < v[i2]; });
    }
    template void argsort_f<int>(vector<int> &v, vector<int> &ind);
    template void argsort_f<double>(vector<double> &v, vector<int> &ind);

    double maximum_2Dvec(std::vector<std::vector<double>> &Vec){
        int N = Vec.size();
        std::vector<double> tmp(N, 0);

        for(int i = 0; i < N; i++){
            tmp[i] = *std::max_element(Vec[i].begin(), Vec[i].end());
        }

        double ret = *std::max_element(tmp.begin(), tmp.end());
        return ret;
    }


    template<typename T>
    void write_vec(std::string name, std::vector<T> &y){
        std::fstream in_f;
        in_f.open(name.c_str(), std::ios::out | std::ios::binary);
    //	in_f.write((char*)y, sizeof(T)*N);
        for(T &ele: y) in_f.write((char *)&ele, sizeof(T)) ;
        in_f.close();
    }
    template void write_vec<int>(std::string name, std::vector<int> &y);
    template void write_vec<double>(std::string name, std::vector<double> &y);

};
