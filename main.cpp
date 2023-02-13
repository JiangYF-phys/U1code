#include <iostream>
#include "DMRG.hpp"
#include "measure.hpp"

using namespace std;
using namespace chrono;

using namespace spinless;
using namespace spinless::armchairlatt;
using namespace armchair;

void readpara();
void measure();

int main() {
    cublasCreate(&GlobalHandle);
    // cudaStreamCreate(&BlasStream);
    // cublasSetStream(GlobalHandle, BlasStream);
    cusolverDnCreate(&GlobalDnHandle);
    cudaEventCreate(&cstart);
    cudaEventCreate(&cstop);
    readpara();
    auto start=high_resolution_clock::now();
    
    if (continueflag==0) {
        iter_control(lastsweep);
        warmup();
        LtoR( middleid, ltot-truncpoint-1, true, false);
        RtoL( ltot-truncpoint-1, truncpoint+1, true, false);
        lastsweep=0;
    }
    
    if (continueflag==1 || continueflag==2) {
        nontrunc();
        reconstruct();
        readwave();
        
        // if (continueflag == 3) flipwave();

        if (lastsweep<=sweep) {
            iter_control(lastsweep);
            if (continueflag==1) {
                LtoR( beginid+1, ltot-sweepend-1, false, true);
                RtoL( ltot-sweepend-1, sweepend+1, false, false);
            }

            if (continueflag==2) {
                RtoL( beginid, sweepend+1, false, true);
            }
        } else if (beginid < stopid) {
            iter_control(lastsweep);
            if (continueflag==1) {
                LtoR( beginid+1, stopid, false, true);
            }
            return 0;
        } else if (beginid > stopid) {
            if (continueflag==2) {
                RtoL( beginid, stopid+1, false, true);
            }
            return 0;
        }
    }
    
    string name="out/energy.dat";
    ofstream out;
    for (int i=lastsweep; i<sweep; ++i) {
        cout << "----------------" << endl  << "sweep=" << i+1 << endl << "----------------" << endl;
        out.open(name.c_str(), ios::out | ios::app);
        if (out.is_open()) {
            out << endl << "----------------" << endl << "  sweep=" << i+1 << endl  << "----------------" << endl;
            out.close();
        }
        iter_control(i);
        LtoR( sweepend+1, ltot-sweepend-1, false, false);
        RtoL( ltot-sweepend-1, sweepend+1, false, false);
    }
    
    if (lastsweep<=sweep) {
        cout << "----------------" << endl << "sweep=" << sweep+1 << endl << "----------------" << endl;
        out.open(name.c_str(), ios::out | ios::app);
        if (out.is_open()) {
            out << endl << "----------------" << endl << "  sweep=" << sweep+1 << endl << "----------------" << endl;
            out.close();
        }
        iter_control(sweep);
        LtoR( sweepend+1, stopid, false, false);
    }
    
    out.open("out/info", ios::out | ios::app);
    if (out.is_open()) {
        out << "lastsweep  = " << sweep+1 << " is # of sweep we stop at; lastsweep-1 has finished." << endl;
        out.close();
    }
    savewave();
    out.open("stopinfo", ios::out | ios::trunc);
    if (out.is_open()) {
        out << "stop as expected" << endl;
    }
    auto stop=high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    cout << endl << "total time=" << duration.count() << "s" << endl;
    name="out/energy.dat";
    out.open(name.c_str(),ios::app);
    if (out.is_open()) {
        out << endl << "total time=" << duration.count() << "s" << endl;
        out.close();
    }
    
    if (measureflag==1) measure();

	delete [] mybase;
    ground.clear();
    site.clear();
    siteCPU.clear();

    cusolverDnDestroy(GlobalDnHandle);
    cublasDestroy(GlobalHandle);
    cudaEventDestroy(cstart);
    cudaEventDestroy(cstop);
    cudaDeviceReset();
	cout << "END" << endl;
    
    return 0;
}

void readpara() {
    ifstream in("info", ios::in);
	ofstream out("out/info", ios::out | ios::trunc); out << scientific;
	readhelp(in, out); readother(in, out);
	out.close(); in.close();

	s_z = define_sz();
    s_p = define_sp(); 
	fc_u = define_fc_u(); 
    fc_d = define_fc_d();
	fc_u_d = fc_u.conj(0); 
    fc_d_d = fc_d.conj(0); 
	fn = define_fn();
	site = define_site(0.0);
    fn.toCPU(fn_CPU, 0);
    s_z.toCPU(sz_CPU, 0);
    site.toCPU(siteCPU);

	vector<bond> latt = mylattice();

	setbase(latt);
	Jtarget = define_J(0);
	if (continueflag == 0) lastsweep = 0;
	if (refpoint == 0) refpoint = int((lx + 1) * 3 / 4) - 1;

	train.clear();
	for (int i = 0; i < ltot+1; i++) { train.push_back(0);}
	if (trainflag == 1) {
        // for (int j = 0; j < ly; j++) {
        //     train[tolat(0, j)+1]=V_1*ntot/ltot;
        //     train[tolat(lx-1, j)+1]=V_1*ntot/ltot;
        // }
        for (int i = 0; i < lx/3; i++) {
            for (int j = 0; j < ly; j++) {
                train[tolat(3*i+1, j)+1]= - 0.5;
            }
        }
	}
    train_site.clear();
    for (int i = 0; i < ltot+1; i++) { train_site.push_back(define_site(train[i]) );}

    out.open(file+"runflag", ios::out | ios::trunc);
    int i=1;
    out << i << endl;
    out.close();
}

void measure() {
	cout << "measure" << endl;
	// sys_trun = new reducematrix[stopid]();
	// env_trun = new reducematrix[ltot - stopid]();

	// for (int i = truncpoint + 1; i < stopid; ++i) {
	// 	sys_trun[i].fromdisk(file + "systrun" + to_string(i), 'n');
	// }
	// for (int i = truncpoint + 1; i < ltot - stopid; ++i) {
	// 	env_trun[i].fromdisk(file + "envtrun" + to_string(i), 'n');
	// }

    twopoint=settwopoint(int(ly/2.0));

    ofstream out;
    string name="out/energy.dat";
    out.open(name.c_str(),ios::app);
    fourpoint=setfourpoint(int(ly/2.0), out);
    out.close();
    
    cudaStream_t stream[2];
    for (size_t j = 0; j < 2; j++) {
        cudaStreamCreate(&stream[j]);
    }
    cublasSetStream(GlobalHandle, stream[0]);

	measurehelp(stream);

    for (size_t i = 0; i < 2; i++) {
        cudaStreamDestroy(stream[i]);
    }

	// for (int i = truncpoint + 1; i < stopid; ++i) {
	// 	sys_trun[i].clear();
	// }
	// for (int i = truncpoint + 1; i < ltot - stopid; ++i) {
	// 	env_trun[i].clear();
	// }
	// delete[] sys_trun; delete[] env_trun;
	// sys_trun = NULL; env_trun = NULL;
}