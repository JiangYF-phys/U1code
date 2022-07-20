#ifndef lattice_hpp
#define lattice_hpp

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

struct bond {
public:
    int l1, l2;
    vector<double> amp;
    
    bond(vector<double> amp, vector<double> coef = {}, int l1 = 0, int l2 = 0);
    ~bond();
    void def(vector<double> amp, vector<double> coef, int l1, int l2);
};

ostream& operator <<(ostream& out, const vector<bond>& lattice);

void makelattice(vector<bond> &latt);

struct correlator {
public:
	char labs[3];
	std::vector<int> pts;

	correlator(int l = 0) { pts.resize(l); };
	~correlator() {};
	//correlator& operator =(const correlator& rhs);
};

struct fourblock {
public:
	int sys_len, env_len;
    vector<int> sys_idx, env_idx;
    vector<bond> sys_st1, sys_st2, st1_st2, sys_env, st1_env, st2_env;

    fourblock();
	~fourblock();
	void set(int l1, int ltot, const vector<bond> &lattice, int seq);
};

vector<int> sys_env_helper(const vector<bond> &sys_env);

#endif /* lattice_hpp */
