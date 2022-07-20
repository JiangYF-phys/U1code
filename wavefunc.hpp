#ifndef wavefunc_hpp
#define wavefunc_hpp

#include <stdio.h>
#include <algorithm>
#include "reducematrix.hpp"

struct mul_store {
public:
    int mul_i, mul_j, mul_nl, mul_nr, mul_il, mul_ir, mul_jl, mul_jr;
    double mul_9j, mul_mat;
    
    mul_store(int mul_i, int mul_j, int mul_nl, int mul_nr, int mul_il, int mul_ir, int mul_jl, int mul_jr, double mul_9j, double mul_mat);
    ~mul_store();
};

class wave : public reducematrix {
public:
    using reducematrix::reducematrix;
    // wave() {};
    // ~wave() {};
    double norm() const;
    double normalize(cudaStream_t stream);
    double dot(const wave &block1) const;
    
    void initial(const vector<repmap> &sys_map, const vector<repmap> &env_map, int snum, int fnum);
    void mul_help(const reducematrix &block1, const reducematrixCPU &block2, const mblock &myw, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], double* tmp_mat, const vector<repmap> &mapl, const vector<repmap> &mapr, const vector<mul_store> &sys_store, const vector<mul_store> &env_store, cudaStream_t stream);
    void mul(const reducematrix &block1, const reducematrixCPU &block2, const wave &myw, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &mapl, const vector<repmap> &mapr, cudaStream_t stream);
    void transLtoR(const wave &myw, const reducematrix &systrun, const reducematrix &envtrun, const vector<repmap> sysmap, const vector<repmap> envmap);
    void transRtoL(const wave &myw, const reducematrix &systrun, const reducematrix &envtrun, const vector<repmap> sysmap, const vector<repmap> envmap);
    
    
//lagacy
public:
    void mul(const reducematrix &block1, const wave &myw);
    void mul(const wave &myw, const reducematrix &block2);
    void mul(const reducematrix &block1, const wave &myw, const reducematrix &block2, const double &para, const char &flag1, const char &flag2);
};


class wave_CPU {
    public:
    vector<mblockCPU*> mat;
    double* val;
    int tot_size;

    public:
    wave_CPU();
    ~wave_CPU();
    void clear();
    void construct(const wave &myw);
    void setzero();
    // void constructGPU(wave &myw);
    void copy(const wave &myw, cudaStream_t stream);
    // void copyGPU(wave &myw, cudaStream_t stream);
    // void fromGPU(const wave &myw, cudaStream_t stream);
    void toGPU(wave &myW, cudaStream_t stream) const;
    void mul_num(const double& alpha);
    void mul_add(const double& alpha, const wave_CPU& wav);
    double mem_size();
    void copyval(int loc, const mblock &myw, cudaStream_t stream);
    mblock toGPU(int i, cudaStream_t stream) const;
    double dot(const wave &myw, cudaStream_t stream) const;
    double normalize();
    int search(const int &jleft, const int &jright, const int &nleft, const int &nright) const;
    void mul_help(const reducematrix &block1, const reducematrixCPU &block2, const mblock &myw, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], double* tmp_mat, const vector<repmap> &mapl, const vector<repmap> &mapr, const vector<mul_store> &sys_store, const vector<mul_store> &env_store, cudaStream_t stream);
};

void mul_CtoG(const reducematrix &block1, const reducematrixCPU &block2, const wave_CPU &myw, wave &waveG, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &mapl, const vector<repmap> &mapr, cudaStream_t stream);
void mul_GtoC(const reducematrix &block1, const reducematrixCPU &block2, const wave &myw, wave_CPU &waveC, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &mapl, const vector<repmap> &mapr, cudaStream_t stream);

#endif /* wavefunc_hpp */
