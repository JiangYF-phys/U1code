#ifndef reducematrix_hpp
#define reducematrix_hpp

#include <stdio.h>
#include "mblock.hpp"
#include "repmap.hpp"

class reducematrixCPU {
public: // should be improved
    vector<mblockCPU*> mat;
    int mysign;
    
public:
    reducematrixCPU(int size=0, int sign=1);
    ~reducematrixCPU();
    int size() const;
    int sign() const;
    void sign(int sig);
    void reserve(int size);
    const mblockCPU& get(int i) const;
    void clear();
    void add(const mblockCPU &block);
    void fromdisk(ifstream& in, const char& flag);
    void fromdisk(const string& in, const char& flag);
    double mem_size() const;

    friend ostream& operator <<(ostream& out, const reducematrixCPU& block);

};


class reducematrix {
private:
    vector<mblock*> mat;
    int mysign;
    
public:
    reducematrix(int size=0, int sign=1);
    reducematrix(const reducematrix &rhs);
    reducematrix(const reducematrix &rhs, cudaStream_t stream);
    ~reducematrix();
    int size() const;
	int sign() const;
	void sign(int sig);
    int jmax() const;
    int nmax() const;
    void add(const mblock &block);
    void addC(mblock* block);
    void changemat(const double* val, cudaStream_t stream);
    void setzero(cudaStream_t stream);
    void setran();
    void set(const reducematrix &rhs, cudaStream_t stream);
    void toidentity();
    void clear();
    void fromC(const reducematrixCPU &rhs, cudaStream_t stream);
    void toCPU(reducematrixCPU &rhs, cudaStream_t stream) const;
    const mblock& get(int i) const;
    void num_mul_block(int i, double num, cudaStream_t stream);
    void num_mul(double num, cudaStream_t stream);
    void mul_add(const double& alpha, const reducematrix& block, cudaStream_t stream);
    void mconjtoblock(int i, double num);
    void setblockpart(int i, int partA[4], const mblock& block, int partB[4], cudaStream_t stream);
    void addsubblock(int loc, int bgn1, int bgn2, int len1, int len2, const mblock& part, cudaStream_t stream);
    void mult_subblock_subblock_rank(int loc, const double alpha, const mblock &block1, const double b2mat, const mblock &myw, const double b3mat, const mblock &block4, double* tmp_mat, const int bgn[4],const char flag[4],cudaStream_t stream);
    void todisk(ofstream& out, const char& flag) const;
    void todisk(const string& out, const char& flag) const;
    void fromdisk(ifstream& in, const char& flag, cudaStream_t stream);
    void fromdisk(const string& in, const char& flag, cudaStream_t stream);
    void trunc(const reducematrix &trunc);
	const int& getjr(int i) const;
	const int& getnr(int i) const;
	const int& getsr(int i) const;
	const int& getjl(int i) const;
	const int& getnl(int i) const;
	const int& getsl(int i) const;
    double mem_size() const;
    
    
public:
    reducematrix& operator =(const reducematrix &rhs);
    reducematrix& operator +=(const reducematrix &block1);
	int search(const int &jleft, const int &jright, const int &nleft, const int &nright) const;
    void prod(const reducematrix &block1, const reducematrix &block2, const vector<repmap> &map, double coef, const char &flag1, const char &flag2);
    void prod_id(const reducematrix &block1, const reducematrix &block2, const vector<repmap> &map, const double &coef, const char &id_pos, cudaStream_t stream);
    reducematrix conj(cudaStream_t stream) const;
    reducematrix applytrunc(const reducematrix &trunc, cudaStream_t stream);

public:
    friend ostream& operator <<(ostream& out, const reducematrix& block);
    friend reducematrix operator *(const double &num, const reducematrix &block1);
    friend vector<repmap> jmap(const reducematrix &block1, const reducematrix &block2);
    
    
//lagacy
public:
    //only in measurement
    reducematrix mul(const reducematrix &block1) const;
    void mult_block_rank(int loc, const double alpha, const mblock &block1, const mblock &block2, char flag1, char flag2);
    void mult_block_block_rank(int loc, const double alpha, const mblock &block1, const mblock &wave, const mblock &block2, char flag1, char flag2);
};

#endif /* reducematrix_hpp */
