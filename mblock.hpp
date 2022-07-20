#ifndef mblock_hpp
#define mblock_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cstring>
#include <chrono>
#include "mkl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
//#define NDEBUG
#include <assert.h>

using namespace std;
using namespace chrono;
// subblock

struct mblockCPU {
public:
    double *mat;
    int jleft, jright, sleft, sright, nleft, nright;
    
    mblockCPU(const int& jleft=0, const int& jright=0, const int& sleft=0, const int& sright=0, const int& nleft=0, const int& nright=0);
    mblockCPU(const mblockCPU& block);
	mblockCPU(ifstream& in, const char& flag);
    ~mblockCPU();
    void clear();
    void mconj(const double alpha);
};

ostream& operator <<(ostream& out, const mblockCPU &block);

struct mblock {
public:
    double *mat;
    int jleft, jright, sleft, sright, nleft, nright;
    
	mblock(const mblock& block, cudaStream_t stream);
    mblock(const mblock& block);
    mblock(const mblockCPU& block, cudaStream_t stream);
    mblock(const int& jleft=0, const int& jright=0, const int& sleft=0, const int& sright=0, const int& nleft=0, const int& nright=0);
	mblock(ifstream& in, const char& flag, cudaStream_t stream);
    ~mblock();
    void todisk(ofstream& out, const char& flag) const;
	void set(int jleft, int jright, int nleft, int nright);
	void set(const mblock& block);
    void mconj(const double alpha, cudaStream_t stream);
    mblock& operator =(const mblock &rhs);
    void mul_num(const double alpha, cudaStream_t stream);
    // mblock& operator *=(const double alpha);
	mblock operator *(const mblock &block1) const;
    void fromC(const mblockCPU &blockC);
    void toCPU(mblockCPU &blockC);
	mblock block(int bgn1, int bgn2, int len1, int len2);
	double norm() const;
	void AddProd(double alpha, const mblock &block1, const mblock &block2, const int bgn1, const int bgn2, char flag1, char flag2);
    void Add_Prod_Id(const double& alpha, const mblock &block1, const int& id_size, const int& bgn1, const int& bgn2, const char& id_pos, cudaStream_t stream);
    void addto(const mblock &block1, cudaStream_t stream);
    void mult_subblock_subblock_rank(const double alpha, const mblock &block1, const double b2mat, const mblock &myw, const double b3mat, const mblock &block4, double* tmp_mat, const int bgn[4],const char flag[4],cudaStream_t stream);
    void mult(const double alpha, const mblock &block1, const mblock &block2, char flag1, char flag2);
//legacy
    mblock& operator +=(const mblock &block1);
};

__global__ void matAdd(double alpha, double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2);
__global__ void matcopypart(double alpha, double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2);

bool checktime(const mblock &block1, const mblock &block2, char flag1, char flag2);
bool checkplus(const mblock &block1, const mblock &block2, char flag);

mblock mconj(const mblock &block1, const double alpha);

//mblock operator *(const double &num, const mblock &block1);
mblock multwithrank(const double alpha, const mblock &block1, const mblock &block2, char flag1, char flag2);

ostream& operator <<(ostream& out, const mblock &block);

#endif /* mblock_hpp */
