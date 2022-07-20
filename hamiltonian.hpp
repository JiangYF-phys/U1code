#ifndef hamiltonian_hpp
#define hamiltonian_hpp

#include <stdio.h>
#include "reducematrix.hpp"

struct HamiltonCPU {
public:
    reducematrixCPU Ham;
    int myl;
    vector<char> stype;
    vector<int> opl;
    reducematrixCPU **op;

public:
    HamiltonCPU(int l=0, vector<int> opl = vector<int>(0,0), vector<char> stype = vector<char>(0, 'n'));
    HamiltonCPU(const string filename);
    ~HamiltonCPU();
    void clear();
    void fromdisk(const string filename);
    double mem_size() const;
};

struct Hamilton {
public:
    reducematrix Ham;
    int myl;
    vector<char> stype;
    vector<int> opl;
    reducematrix **op;

public:
    Hamilton(int l=0, vector<int> opl = vector<int>(0,0), vector<char> stype = vector<char>(0, 'n'));
    Hamilton(const Hamilton& ham);
    Hamilton(const string filename);
    ~Hamilton();

public:
	int len() const;
    double mem_size() const;
    
public:
    Hamilton& operator =(const Hamilton &rhs);
    void clear();
    void fromdisk(const string filename);
    void fromC(const HamiltonCPU &hamC);
    void toCPU(HamiltonCPU &hamC);
    void optodisk(const string filename) const;
	void truncHam(const reducematrix &trunc);
};


reducematrix basistoid(const vector<repmap> basis, char lr);

#endif /* hamiltonian_hpp */
