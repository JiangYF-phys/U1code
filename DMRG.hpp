#ifndef DMRG_hpp
#define DMRG_hpp

#include <stdio.h>
#include <string>
#include <cstdio>
#include <thread>
#include "extend.hpp"
#include "lanczos.hpp"
#include "truncation.hpp"

void nontrunc();
void warmup();
void reconstruct();
void LtoR(const int &beg, const int &end, const bool &initial,const bool &continu);
void RtoL(const int &beg, const int &end, const bool &initial);
void savewave();
void readwave();
void flipwave();

#endif /* DMRG_hpp */
