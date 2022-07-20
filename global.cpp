#include "global.hpp"
#include <string>

vector<fourblock> *mybase;
wave ground;
Hamilton Jtarget, site;
HamiltonCPU siteCPU;

reducematrix s_z, s_p, fc_u, fc_d, fc_u_d, fc_d_d, fn;
reducematrixCPU fn_CPU, sz_CPU;

int num_op;

double j_1, j_2, j_3, hop, hopp, hubU, V_1, V_2, lan_error, trun_error, lan_error_0, trun_error_0;
int lx, ly, ntot, jtot, sweep, lastsweep, kept_min, kept_max, truncpoint, ltot, middleid, stopid, beginid, refpoint;
int continueflag, measureflag, trainflag, constructflag, bshift;
int lanczos_ver, adjustflag;
vector<int> twopoint;
vector<vector<bond>> fourpoint;
vector<double> def_int, icoef, train;
vector<char> optype, savetype;
// reducematrix *sys_trun, *env_trun;

cublasHandle_t GlobalHandle;
cusolverDnHandle_t GlobalDnHandle;
cudaStream_t BlasStream;
double time_1, time_2, time_3, time_4; 
cudaEvent_t cstart, cstop;

template <typename T>
void infoio(ifstream& myin, ofstream& myout, T& item) {
	string name;
	getline(myin, name, '='); myin >> item;
	myout << name << "= " << item;
}

namespace spinless {
	void readhelp(ifstream& in, ofstream& out) {
		infoio(in, out, hop);
		infoio(in, out, hopp);
		infoio(in, out, V_1);
		infoio(in, out, V_2);
		infoio(in, out, lx);
		infoio(in, out, ly);
		infoio(in, out, ntot);
		infoio(in, out, jtot);
	}

	Hamilton define_site(double mmu) {
		num_op=3;
		adjustflag=0;
		optype.resize(num_op); savetype.resize(num_op);
		def_int.resize(num_op); icoef.resize(num_op);
		optype[0] = 'c'; savetype[0] = 'n'; def_int[0] = -hop; icoef[0] = 1;
		optype[1] = 'n'; savetype[1] = 'u'; def_int[1] = V_1 ; icoef[1] = 1;

	    vector<int> l(num_op,1);
    	Hamilton myh(1,l, savetype);
	    reducematrix h0(0,1);
		double *va=new double[1];

		mblock a(0,0,1,1,0,0);
		va[0]=0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(a);
		
		mblock b(0,0,1,1,1,1);
		va[0]=0;//-mmu;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(b);

		myh.Ham=h0;
		myh.op[0][0] = fc_u;
		myh.op[1][0] = fn;

		delete [] va;
		return myh;
	}

	reducematrix define_sz() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(-1, -1, 1, 1, 1, 1);
		va[0] = -0.5;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);
		
		mblock b(1, 1, 1, 1, 1, 1);
		va[0] = 0.5;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(b);

		delete [] va;
		return myspin;
	}

	reducematrix define_sp() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(1, -1, 1, 1, 1, 1);
		va[0] = 1.0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);

		delete [] va;
		return myspin;
	}

	reducematrix define_fn() {
		reducematrix myn(0, 1);
		double *va=new double[1];

		mblock a(0, 0, 1, 1, 0, 0);
		va[0] = 0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(a);
		
		mblock b(0, 0, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(b);

		delete [] va;
		return myn;
	}

	reducematrix define_fc_u() {
		reducematrix myfc(0, -1);
		double *va=new double[1];

		mblock b(0, 0, 1, 1, 0, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(b);

		delete [] va;
		return myfc;
	}
	
	reducematrix define_fc_d() {
		reducematrix myfc(0, -1);
		double *va=new double[1];

		mblock b(0, 0, 1, 1, 0, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(b);

		delete [] va;
		return myfc;
	}

	namespace squarelatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 1) {
						loc2 = tolat(i + 1, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = -hopp; amp[1] = V_2; 
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					loc2 = tolat(i + 1, j - 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}

	namespace brickwalllatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					if (i%2==1) {
						int loc1 = tolat(i, j);
						int loc2 = tolat(i - 1, j + 1);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = -hopp; amp[1] = V_2; 
			for (int i = 0; i < lx - 2; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 2, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j+1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			for (int i = 2; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i-2, j+1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}

	namespace armchairlatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					if ((i+j)%2==0) {
						int loc1 = tolat(i, j);
						int loc2 = tolat(i, j + 1);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = -hopp; amp[1] = V_2; 
			for (int i = 0; i < lx - 2; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 2, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			for (int i = 0; i < lx-1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i+1, j+1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			for (int i = 0; i < lx-1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i+1, j-1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}
}

namespace Hubbard {
	void readhelp(ifstream& in, ofstream& out) {
		infoio(in, out, hop);
		infoio(in, out, hopp);
		infoio(in, out, hubU);
		infoio(in, out, V_1);
		infoio(in, out, lx);
		infoio(in, out, ly);
		infoio(in, out, ntot);
		infoio(in, out, jtot);
	}

	Hamilton define_site(double mmu) {
		num_op=3;
		adjustflag=1;
		optype.resize(num_op); savetype.resize(num_op);
		def_int.resize(num_op); icoef.resize(num_op);
		optype[0] = 'c'; savetype[0] = 'n'; def_int[0] = -hop; icoef[0] = 1;
		optype[1] = 'c'; savetype[1] = 'n'; def_int[1] = -hop; icoef[1] = 1;
		optype[2] = 'n'; savetype[2] = 'u'; def_int[2] = V_1 ; icoef[2] = 1;

	    vector<int> l(num_op,1);
    	Hamilton myh(1,l, savetype);
	    reducematrix h0(0,1);
		double *va=new double[1];

		mblock a(0,0,1,1,0,0);
		va[0]=0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(a);
		
		mblock b(1,1,1,1,1,1);
		va[0]=0;//-mmu;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(b);
		
		mblock c(-1,-1,1,1,1,1);
		va[0]=0;//-2*mmu;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(c);
		
		mblock d(0,0,1,1,2,2);
		va[0]=hubU;//-2*mmu;
		cudaMemcpy(d.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(d);

		myh.Ham=h0;
		myh.op[0][0] = fc_u;
		myh.op[1][0] = fc_d;
		myh.op[2][0] = fn;

		delete [] va;
		return myh;
	}

	reducematrix define_sz() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(-1, -1, 1, 1, 1, 1);
		va[0] = -0.5;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);
		
		mblock b(1, 1, 1, 1, 1, 1);
		va[0] = 0.5;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(b);

		delete [] va;
		return myspin;
	}

	reducematrix define_sp() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(1, -1, 1, 1, 1, 1);
		va[0] = 1.0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);

		delete [] va;
		return myspin;
	}

	reducematrix define_fn() {
		reducematrix myn(0, 1);
		double *va=new double[1];

		mblock a(0, 0, 1, 1, 0, 0);
		va[0] = 0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(a);
		
		mblock b(-1, -1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(b);
		
		mblock c(1, 1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(c);
		
		mblock d(0, 0, 1, 1, 2, 2);
		va[0] = 2;
		cudaMemcpy(d.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(d);

		delete [] va;
		return myn;
	}

	reducematrix define_fc_u() {
		reducematrix myfc(0, -1);
		double *va=new double[1];

		mblock b(0, 1, 1, 1, 0, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(b);

		mblock c(-1, 0, 1, 1, 1, 2);
		va[0] = 1;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(c);

		delete [] va;
		return myfc;
	}
	
	reducematrix define_fc_d() {
		reducematrix myfc(0, -1);
		double *va=new double[1];

		mblock b(0, -1, 1, 1, 0, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(b);

		mblock c(1, 0, 1, 1, 1, 2);
		va[0] = -1;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(c);

		delete [] va;
		return myfc;
	}

	namespace squarelatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 1) {
						loc2 = tolat(i + 1, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = -hopp; amp[1] = -hopp; amp[2] = 0; 
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					loc2 = tolat(i + 1, j - 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}

	namespace squarelattOBC {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 1);
					if (j < ly - 1) {
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}

					if (i < lx - 1) {
						loc2 = tolat(i + 1, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = -hopp; amp[1] = -hopp; amp[2] = 0; 
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly-1; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}

				for (int j = 1; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j - 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}
}

namespace tJmodel {
	void readhelp(ifstream& in, ofstream& out) {
		infoio(in, out, hop);
		infoio(in, out, hopp);
		infoio(in, out, j_1);
		infoio(in, out, j_2);
		infoio(in, out, lx);
		infoio(in, out, ly);
		infoio(in, out, ntot);
		infoio(in, out, jtot);
	}

	Hamilton define_site(double mmu) {
		num_op=5;
		adjustflag=1;
		optype.resize(num_op); savetype.resize(num_op);
		def_int.resize(num_op); icoef.resize(num_op);
		optype[0] = 'c'; savetype[0] = 'n'; def_int[0] = -hop; icoef[0] = 1;
		optype[1] = 'c'; savetype[1] = 'n'; def_int[1] = -hop; icoef[1] = 1;
		optype[2] = 'n'; savetype[2] = 'u'; def_int[2] =  j_1; icoef[2] = 1;
		optype[3] = 'c'; savetype[3] = 'n'; def_int[3] =j_1/2; icoef[3] = 1;
		optype[4] = 'n'; savetype[4] = 'u'; def_int[4] =-j_1/4; icoef[4] = 1;

	    vector<int> l(num_op,1);
    	Hamilton myh(1,l, savetype);
	    reducematrix h0(0,1);
		double *va=new double[1];

		mblock a(0,0,1,1,0,0);
		va[0]=0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(a);
		
		mblock b(1,1,1,1,1,1);
		va[0]=0;//-mmu;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(b);
		
		mblock c(-1,-1,1,1,1,1);
		va[0]=0;//-2*mmu;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(c);

		myh.Ham=h0;
		myh.op[0][0] = fc_u;
		myh.op[1][0] = fc_d;
		myh.op[2][0] = s_z;
		myh.op[3][0] = s_p;
		myh.op[4][0] = fn;

		delete [] va;
		return myh;
	}

	reducematrix define_sz() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(-1, -1, 1, 1, 1, 1);
		va[0] = -0.5;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);
		
		mblock b(1, 1, 1, 1, 1, 1);
		va[0] = 0.5;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(b);

		delete [] va;
		return myspin;
	}

	reducematrix define_fn() {
		reducematrix myn(0, 1);
		double *va=new double[1];

		mblock a(0, 0, 1, 1, 0, 0);
		va[0] = 0;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(a);
		
		mblock b(-1, -1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(b);
		
		mblock c(1, 1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(c);

		delete [] va;
		return myn;
	}

	reducematrix define_sp() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(1, -1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);

		delete [] va;
		return myspin;
	}

	reducematrix define_fc_u() {
		reducematrix myfc(0, -1);
		double *va=new double[1];

		mblock b(0, 1, 1, 1, 0, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(b);

		delete [] va;
		return myfc;
	}
	
	reducematrix define_fc_d() {
		reducematrix myfc(0, -1);
		double *va=new double[1];

		mblock b(0, -1, 1, 1, 0, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myfc.add(b);

		delete [] va;
		return myfc;
	}

	namespace squarelatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 1) {
						loc2 = tolat(i + 1, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = -hopp; amp[1] = -hopp; amp[2] = j_2; amp[3] = j_2/2; amp[4] = -j_2/4;
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					loc2 = tolat(i + 1, j - 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}

}

namespace Heisenberg {
	void readhelp(ifstream& in, ofstream& out) {
		infoio(in, out, j_1);
		infoio(in, out, j_2);
		infoio(in, out, j_3);
		infoio(in, out, lx);
		infoio(in, out, ly);
		infoio(in, out, ntot);
		infoio(in, out, jtot);
	}

	Hamilton define_site(double mmu) {
		num_op=2;
		adjustflag=1;
		optype.resize(num_op); savetype.resize(num_op);
		def_int.resize(num_op); icoef.resize(num_op);
		optype[0] = 'n'; savetype[0] = 'u'; def_int[0] =  j_1; icoef[0] = 1;
		optype[1] = 'c'; savetype[1] = 'n'; def_int[1] =j_1/2; icoef[1] = 1;

	    vector<int> l(num_op,1);
    	Hamilton myh(1,l, savetype);
	    reducematrix h0(0,1);
		double *va=new double[1];
		
		mblock b(1,1,1,1,1,1);
		va[0]=0;//-mmu;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(b);
		
		mblock c(-1,-1,1,1,1,1);
		va[0]=0;//-2*mmu;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		h0.add(c);

		myh.Ham=h0;
		myh.op[0][0] = s_z;
		myh.op[1][0] = s_p;

		delete [] va;
		return myh;
	}

	reducematrix define_sz() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(-1, -1, 1, 1, 1, 1);
		va[0] = -0.5;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);
		
		mblock b(1, 1, 1, 1, 1, 1);
		va[0] = 0.5;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(b);

		delete [] va;
		return myspin;
	}

	reducematrix define_fn() {
		reducematrix myn(0, 1);
		double *va=new double[1];
		
		mblock b(-1, -1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(b);
		
		mblock c(1, 1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(c.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myn.add(c);

		delete [] va;
		return myn;
	}

	reducematrix define_sp() {
		reducematrix myspin(0, 1);
		double *va=new double[1];
		
		mblock a(1, -1, 1, 1, 1, 1);
		va[0] = 1;
		cudaMemcpy(a.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
		myspin.add(a);

		delete [] va;
		return myspin;
	}

	reducematrix define_fc_u() {
		reducematrix myfc(0, -1);
		return myfc;
	}
	
	reducematrix define_fc_d() {
		reducematrix myfc(0, -1);
		return myfc;
	}

	namespace squarelatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 1) {
						loc2 = tolat(i + 1, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0]=j_2; amp[1] = j_2/2;
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					loc2 = tolat(i + 1, j - 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}

	namespace trianglelatt {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx * ly;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 1) {
						loc2 = tolat(i + 1, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);

						loc2 = tolat(i + 1, j-1);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}

			amp[0] = j_2; amp[1] = j_2/2;
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i + 1, j - 2);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					loc2 = tolat(i + 1, j + 1);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 2) {
						loc2 = tolat(i + 2, j - 1);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}
			
			amp[0] = j_3; amp[1] = j_3/2;
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i, j + 2);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);

					if (i < lx - 2) {
						loc2 = tolat(i + 2, j);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);

						loc2 = tolat(i + 2, j - 2);
						tmp.def(amp, icoef, loc1, loc2);
						latt.push_back(tmp);
					}
				}
			}
			return latt;
		}
	}

	namespace chain {
		vector<bond> mylattice() {
			vector<bond> latt;
			ltot = lx;
			vector<double> amp{def_int.begin(), def_int.end()};
			bond tmp(amp, icoef, 0, 0);
			for (int i = 0; i < lx - 1; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i+1, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			amp[0]=j_2; amp[1] = j_2/2;
			for (int i = 0; i < lx - 2; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i+2, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}

			amp[0]=j_3; amp[1] = j_3/2;
			for (int i = 0; i < lx - 3; ++i) {
				for (int j = 0; j < ly; j++) {
					int loc1 = tolat(i, j);
					int loc2 = tolat(i+3, j);
					tmp.def(amp, icoef, loc1, loc2);
					latt.push_back(tmp);
				}
			}
			return latt;
		}
	}

}

namespace square {
	vector<int> settwopoint(int y) {
		vector<int> mysites;
		mysites.clear();
		for (int i = 0; i <= lx/2+1; ++i) {
			mysites.push_back(tolat(refpoint+i, y));
		}
		return mysites;
	}

	vector<vector<bond>> setfourpoint(int y, ofstream& out) {
		vector<vector<bond>> allsets;
		vector<double> amp{def_int.begin(), def_int.end()};
		bond tmp(amp, icoef, 0, 0);
		allsets.clear();

		vector<bond> mysites;
		int loc1, loc2;

		out << "pair correlation label:" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i, y+1);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YX" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i, y+1);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": XY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": XX" << endl;
		out.close();

		return allsets;
	}
}

namespace triangle {
	vector<int> settwopoint(int y) {
		vector<int> mysites;
		mysites.clear();
		for (int i = 0; i <= lx/2+1; ++i) {
			mysites.push_back(tolat(refpoint+i, y));
		}
		return mysites;
	}

	vector<vector<bond>> setfourpoint(int y, ofstream& out) {
		vector<vector<bond>> allsets;
		vector<double> amp{def_int.begin(), def_int.end()};
		bond tmp(amp, icoef, 0, 0);
		allsets.clear();

		vector<bond> mysites;
		int loc1, loc2;

		out << "pair correlation label:" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i, y+1);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YX" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i, y+1);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": XY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/2+1; ++i) {
			loc1 = tolat(refpoint+i, y);
			loc2 = tolat(refpoint+i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": XX" << endl;


		out.close();

		return allsets;
	}
}

namespace brickwall {
	vector<int> settwopoint(int y) {
		vector<int> mysites;
		mysites.clear();
		for (int i = 0; i <= lx/2+1; ++i) {
			mysites.push_back(tolat(refpoint+i, y));
		}
		return mysites;
	}

	vector<vector<bond>> setfourpoint(int y, ofstream& out) {
		vector<vector<bond>> allsets;
		vector<double> amp{def_int.begin(), def_int.end()};
		bond tmp(amp, icoef, 0, 0);
		allsets.clear();

		vector<bond> mysites;
		int loc1, loc2;

		out << "pair correlation label:" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint-1, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y+1);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": ZZ" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint-1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": XX" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint-1, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": ZY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint-1, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": ZX" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YX" << endl;
		out.close();
		return allsets;
	}
}

namespace armchair {
	vector<int> settwopoint(int y) {
		vector<int> mysites;
		mysites.clear();
		for (int i = 0; i <= lx/2+1; ++i) {
			mysites.push_back(tolat(refpoint+i, y));
		}
		return mysites;
	}

	vector<vector<bond>> setfourpoint(int yin, ofstream& out) {
		vector<vector<bond>> allsets;
		vector<double> amp{def_int.begin(), def_int.end()};
		bond tmp(amp, icoef, 0, 0);
		allsets.clear();

		vector<bond> mysites;
		int loc1, loc2;

		out << "pair correlation label:" << endl;
		
		int y=1;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i, y+1);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": ZZ" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint-1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": XX" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i+1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": ZY" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint, y+1);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": ZX" << endl;

		mysites.clear();
		loc1 = tolat(refpoint, y);
		loc2 = tolat(refpoint+1, y);
		tmp.def(amp, icoef, loc1, loc2);
		mysites.push_back(tmp);
		for (int i = 1; i < lx/4+1; ++i) {
			loc1 = tolat(refpoint+2*i, y);
			loc2 = tolat(refpoint+2*i-1, y);
			tmp.def(amp, icoef, loc1, loc2);
			mysites.push_back(tmp);
		}
		allsets.push_back(mysites);
		out << allsets.size() << ": YX" << endl;
		out.close();
		return allsets;
	}
}

void readother(ifstream& in, ofstream& out) {
	infoio(in, out, kept_min);
	infoio(in, out, kept_max);
	infoio(in, out, truncpoint);
	infoio(in, out, lan_error_0);
	infoio(in, out, trun_error_0);
	infoio(in, out, sweep);
	infoio(in, out, beginid);
	infoio(in, out, stopid);
	infoio(in, out, measureflag);
	infoio(in, out, continueflag);
	infoio(in, out, trainflag);
	infoio(in, out, constructflag);
	infoio(in, out, refpoint);
	infoio(in, out, lanczos_ver);
	infoio(in, out, bshift);
	string name;
	getline(in, name, '='); in >> lastsweep;
	out << endl;
}

Hamilton define_J(int myj) {
    vector<int> l(num_op,0);
    Hamilton myh(1,l,savetype);
    reducematrix h0(0,1);
    mblock b(myj,myj,1,1,0,0);
    // b.mat[0]= 0;
	double *va=new double[1];
	va[0]=0;
	cudaMemcpy(b.mat, va, sizeof(double), cudaMemcpyHostToDevice); 
    h0.add(b);
    
    myh.Ham=h0;
	delete [] va;
    return myh;
}

void setbase(vector<bond>& latt) {
	makelattice(latt);
	middleid = int((ltot + 1) / 2.0);
	mybase = new vector<fourblock>[num_op];
	for (int i = 0; i < ltot; ++i) {
		for (int seq = 0; seq < num_op; ++seq) {
			fourblock tmp;
			tmp.set(i, ltot, latt, seq);
			mybase[seq].push_back(tmp);
		}
	}
}

Hamilton makeJsite() {
	vector<int> l(num_op, 1);
	Hamilton newblock(1, l, savetype);

	vector<repmap> basis;

	basis = jmap(Jtarget.Ham, site.Ham);
	newblock.Ham.prod_id(Jtarget.Ham, site.Ham, basis, 1.0, 'r', 0);
	newblock.Ham.prod_id(Jtarget.Ham, site.Ham, basis, 1.0, 'l', 0);

	for (int seq = 0; seq < num_op; ++seq) {
		newblock.op[seq][0].prod_id(Jtarget.Ham, site.op[seq][0], basis, 1.0, 'l', 0);
	}

	return newblock;
}

void iter_control(int nth) {
    if (nth<=1) {
        lan_error=max(lan_error_0, 0.5);
        trun_error=max(trun_error_0, 0.1);
    } else if (nth<=2) {
        lan_error=max(lan_error_0, 0.001);
        trun_error=max(trun_error_0, 0.01);
    } else if (nth<=4) {
        lan_error=max(lan_error_0, 0.0001);
        trun_error=max(trun_error_0, 0.001);
    } else if (nth<=6) {
        lan_error=max(lan_error_0, 0.00001);
        trun_error=max(trun_error_0, 0.0001);
    } else {
        lan_error=lan_error_0;
        trun_error=trun_error_0;
    }
    string name="out/energy.dat";
    ofstream out(name.c_str(), ios::out | ios::app);
    out << scientific;
    if (out.is_open()) {
        out << "trunc_err_0 = " << trun_error <<  ", lancz_err_0 = " << lan_error << endl;
        out.close();
    }
    cout << "trunc_err_0 = " << trun_error << ", lancz_err_0 = " << lan_error << endl;
}
