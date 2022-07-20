#ifndef truncation_hpp
#define truncation_hpp

#include <stdio.h>
#include "global.hpp"

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v);

reducematrix wavetorou(const wave &myw, char side, cudaStream_t stream);
reducematrix routotrunc(const reducematrix &rou, bool spectrumflag);

#endif /* truncation_hpp */
