#ifndef extend_hpp
#define extend_hpp

#include <stdio.h>
#include <algorithm>
#include <string>
#include "global.hpp"

using namespace std;

Hamilton addonesite(const Hamilton &block, vector<repmap> &map, char sore);
void addonesite_replace(Hamilton &block, vector<repmap> &map, const reducematrix &trunc, char sore);
void Htowave(const Hamilton &sys, const wave &trail, wave &newwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, cudaStream_t stream);

void HtowaveCtoG(const Hamilton &sys, const wave_CPU &trail, wave &newwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, cudaStream_t stream);

void HtowaveGtoC(const Hamilton &sys, const wave &trail, wave_CPU &newwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, cudaStream_t stream);


#endif /* extend_hpp */
