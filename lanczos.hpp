#ifndef lanczos_hpp
#define lanczos_hpp

#include <stdio.h>
#include <thread>
#include "global.hpp"
#include "lattice.hpp"
#include "extend.hpp"

wave lanc_main(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis);
void lanc_main_new(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis);
void lanc_main_multi_wave(const Hamilton &sys, wave &lastwave, wave &excited_state, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis);
void lanc_main_V2(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis);
void lanc_main_V3(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, wave_CPU (&wave_store)[2], cudaStream_t stream[2]);
void lanc_main_V4(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, wave_CPU (&wave_store)[4], cudaStream_t stream[2]);

#endif /* lanczos_hpp */
