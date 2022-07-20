#ifndef measure_hpp
#define measure_hpp

#include <stdio.h>
#include "DMRG.hpp"

struct oplist{// to do 
public:
    reducematrix *op;
    int myl;
};

void measuresite(const reducematrix &op, const string &name, cudaStream_t stream[2]);
void measurecorr(const reducematrix &op1, const reducematrix &op2, const double &para, const string &name, cudaStream_t stream[2]);
// void measurefullcorr(const reducematrix &op1, const reducematrix &op2, const double &para,  const string &name);
void measuresccor(const string &name, cudaStream_t stream[2]);
void measuresccor_spinless(const string &name, cudaStream_t stream[2]);

namespace spinless {
    void measurehelp(cudaStream_t stream[2]);
}

namespace Hubbard {
    void measurehelp(cudaStream_t stream[2]);
}

namespace tJmodel {
    void measurehelp(cudaStream_t stream[2]);
}

namespace Heisenberg {
    void measurehelp(cudaStream_t stream[2]);
}

#endif /* measure_hpp */
