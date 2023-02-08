#ifndef global_hpp
#define global_hpp

#include <stdio.h>
#include <math.h>
#include <map>
#include "lattice.hpp"
#include "hamiltonian.hpp"
#include "wavefunc.hpp"

using namespace std;

const int Lan_max_inner=7;
const int Lan_max_outter=24;

struct bond;
struct fourblock;

reducematrix define_spin();
reducematrix define_fc();
Hamilton define_J(int myj);

extern vector<double> def_int, icoef, train;
extern vector<Hamilton> train_site;
extern vector<fourblock> *mybase;
extern Hamilton site, Jtarget;
extern HamiltonCPU siteCPU;
extern wave ground;
extern reducematrix s_z, s_p, fc_u, fc_d, fc_u_d, fc_d_d, fn;
extern reducematrixCPU fn_CPU, sz_CPU;
// extern reducematrix *sys_trun, *env_trun;

extern int num_op;
extern double j_1, j_2, j_3, hop, hopp, hubU, V_1, V_2, lan_error, trun_error, lan_error_0, trun_error_0;
extern int lx, ly, ltot, ntot, jtot, sweep, lastsweep, kept_min, kept_max, truncpoint, sweepend;
extern int middleid, beginid, stopid, continueflag, measureflag, trainflag, constructflag, refpoint, bshift;
extern int lanczos_ver, adjustflag;
extern vector<char> optype, savetype;
extern vector<int> twopoint;
extern vector<vector<bond>> fourpoint;

const string file="temp/";

inline int tolat(int x, int y) { return ly * ((x + 3 * lx) % lx) + (y + 3 * ly) % ly; };
inline int tolat_m(int x, int y) { return tolat(x, y) + 1; };

namespace spinless {
    void readhelp(ifstream& in, ofstream& out);
    Hamilton define_site(double mmu);
    reducematrix define_sz();
    reducematrix define_sp();
    reducematrix define_fn();
    reducematrix define_fc_u();
    reducematrix define_fc_d();
    namespace squarelatt {
        vector<bond> mylattice();
    }
    namespace brickwalllatt {
        vector<bond> mylattice();
    }
    namespace armchairlatt {
        vector<bond> mylattice();
    }
}

namespace Hubbard {
    void readhelp(ifstream& in, ofstream& out);
    Hamilton define_site(double mmu);
    reducematrix define_sz();
    reducematrix define_sp();
    reducematrix define_fn();
    reducematrix define_fc_u();
    reducematrix define_fc_d();
    namespace squarelatt {
        vector<bond> mylattice();
    }
    namespace squarelattOBC {
        vector<bond> mylattice();
    }
}

namespace tJmodel {
    void readhelp(ifstream& in, ofstream& out);
    Hamilton define_site(double mmu);
    reducematrix define_sz();
    reducematrix define_sp();
    reducematrix define_fn();
    reducematrix define_fc_u();
    reducematrix define_fc_d();
    namespace squarelatt {
        vector<bond> mylattice();
    }
    namespace brickwalllatt {
        vector<bond> mylattice();
    }
    namespace armchairlatt {
        vector<bond> mylattice();
    }
}

namespace Heisenberg {
    void readhelp(ifstream& in, ofstream& out);
    Hamilton define_site(double mmu);
    reducematrix define_sz();
    reducematrix define_sp();
    reducematrix define_fn();
    reducematrix define_fc_u();
    reducematrix define_fc_d();
    namespace squarelatt {
        vector<bond> mylattice();
    }
    namespace trianglelatt {
        vector<bond> mylattice();
    }
    namespace chain {
        vector<bond> mylattice();
    }
}

namespace square{
    vector<int> settwopoint(int y);
    vector<vector<bond>> setfourpoint(int y, ofstream& out);
}

namespace triangle{
    vector<int> settwopoint(int y);
    vector<vector<bond>> setfourpoint(int y, ofstream& out);
}

namespace brickwall{
    vector<int> settwopoint(int y);
    vector<vector<bond>> setfourpoint(int y, ofstream& out);
}

namespace armchair{
    vector<int> settwopoint(int y);
    vector<vector<bond>> setfourpoint(int y, ofstream& out);
}

void readother(ifstream& in, ofstream& out);
void setbase(vector<bond>& latt);
Hamilton makeJsite();
void iter_control(int nth);

extern cublasHandle_t GlobalHandle;
extern cusolverDnHandle_t GlobalDnHandle;
extern cudaStream_t BlasStream;
extern cudaEvent_t cstart, cstop;
extern double time_1, time_2, time_3, time_4; 

double physical_memory_used_by_process();
double GPU_memory_used_by_process();

#endif /* global_hpp */
