#include "lanczos.hpp"

using namespace chrono;

wave lanc_main(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis) {
    auto start = high_resolution_clock::now();
    wave* myv = new wave[Lan_max_inner+1]();
    vector<double> alpha, beta;
    wave tmp0(lastwave, 0);
    tmp0.setzero(0);
    myv[0]=tmp0;
    beta.push_back(0); alpha.push_back(0);//skip [0]
    myv[1]=lastwave;// set v[1]
    beta.push_back(0);// set b[1]
    
    double err=1.0, energy=1000.0;
    int i=1;
	vector<double> vec;
    while ( i<Lan_max_inner && err>lan_error ) {
        wave tmp;
        Htowave(sys, myv[i], tmp, env, sys_basis, env_basis, 0);
        // cout << "wave" << endl;
        // cout << tmp << endl;
        alpha.push_back( tmp.dot(myv[i]) ); // a(i)
        // cout << alpha[i] << endl;
        if (i>2) {
			double* diag = new double[alpha.size()-1]();
			double* ndiag= new double[alpha.size()-1]();
			double* dwork= new double[(alpha.size()-1)*(alpha.size()-1)]();
			for (size_t j = 1; j < alpha.size(); ++j) {
				diag[j-1]=alpha[j];
			}
			for (size_t j = 2; j < alpha.size(); ++j) {
				ndiag[j-2]=beta[j];
			}
			LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', alpha.size()-1, diag, ndiag, dwork, alpha.size()-1);
            err=energy-diag[0];
            energy=diag[0];
			delete [] diag; diag=NULL;
			delete [] ndiag; ndiag=NULL;
			vec.clear();
			for (size_t j = 0; j < alpha.size()-1; ++j) {
				vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
			}
			delete [] dwork; dwork=NULL;
        }
        // cout << alpha[i] << ", ";
        tmp.mul_add(-alpha[i], myv[i], 0);
        tmp.mul_add(-beta[i], myv[i-1], 0);
        double nor;
        nor=tmp.normalize(0);
        beta.push_back(nor);
        myv[i+1]=tmp;
        i++;
    }
    // cout << endl;

    wave newwave;
    newwave=myv[0];
    for (size_t i=0; i<alpha.size()-1; ++i) {
        newwave.mul_add(vec[i], myv[i+1], 0);
    }
    delete [] myv; myv=NULL;
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    time_2+=duration.count()/1000.0;
    cout << "Energy=" << energy/ltot << ", iter=" << i-1 << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
    string name="out/energy.dat";
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific << "Energy=" << energy << ", iter=" << i-1 << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
        out.close();
    }
    return newwave;
}

void store_help(const wave &myv, wave_CPU &myvCPU) {
    // if (myvCPU.val==NULL) {
        myvCPU.construct(myv);
    // }
} 

void clear_help(wave_CPU &myvCPU) {
    myvCPU.clear();
}

void lanc_main_new(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis) {
    auto start = high_resolution_clock::now();
    cudaStream_t stream[2];
    for (size_t j = 0; j < 2; j++) {
        cudaStreamCreate(&stream[j]);
    }
    
    int time2=0;
    cublasSetStream(GlobalHandle, stream[0]);
    // wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
    vector<double> alpha, beta;
    vector<double> vec;
    wave myv[2];
    myv[0]=lastwave;
    double err=1.0, energy=1000.0;
    int iout=0, icount=0;
    double Hmem=max(sys.mem_size(),env.mem_size());
    cout << endl;
    cout << "sysmem= " << sys.mem_size()/1024 << "GB, envmem= " << env.mem_size()/1024 << "GB, 3*wavmem= " << 3*lastwave.mem_size()/1024 << "GB" << endl;
    while ( iout<Lan_max_outter && err>lan_error) {
        cout << "outter iter: " << iout+1 << endl;
         // be careful about Lan_max 
        alpha.clear(); beta.clear();
        beta.push_back(0); alpha.push_back(0);//skip [0]
        beta.push_back(0);// set b[1]
        myv[0].setzero(stream[0]);
        myv[1].set(lastwave,stream[0]);
        double memsize=0;
        int upperbound=100;
        int inner=1;
        cout << "inner iter: ";
        wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
        while ( inner < min(upperbound, Lan_max_inner) && err>lan_error ) {
            cout << inner;
        
            auto start2 = high_resolution_clock::now();
            thread th0(store_help, ref(myv[inner%2]), ref(wave_store[inner-1])); // value of reference wave is not important 
            auto stop2 = high_resolution_clock::now();
            auto duration2 = duration_cast<milliseconds>(stop2 - start2);
            time2+=duration2.count();
            Htowave(sys, myv[inner%2], lastwave, env, sys_basis, env_basis, stream[0]);
            alpha.push_back( lastwave.dot(myv[inner%2]) ); // a(i)
            if (inner>1) {
                double* diag = new double[inner]();
                double* ndiag= new double[inner]();
                double* dwork= new double[inner*inner]();
                for (size_t j = 1; j < inner+1; ++j) {
                    diag[j-1]=alpha[j];
                }
                for (size_t j = 2; j < inner+1; ++j) {
                    ndiag[j-2]=beta[j];
                }
                LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', inner, diag, ndiag, dwork, inner);
                err=energy-diag[0];
                cout << ", err=" << err;
                energy=diag[0];
                delete [] diag; diag=NULL;
                delete [] ndiag; ndiag=NULL;
                vec.clear();
                for (size_t j = 0; j < inner; ++j) {
                    vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
                }
                delete [] dwork; dwork=NULL;
            }
            cout << "; " << endl;
            th0.join();
            if (inner==1) {upperbound=int((26000-Hmem)/wave_store[0].mem_size());}
            wave_store[inner-1].copy(myv[inner%2], stream[1]);
            memsize+=wave_store[inner-1].mem_size();

            lastwave.mul_add(-alpha[inner], myv[inner%2], stream[0]);
            lastwave.mul_add(-beta[inner], myv[(inner-1)%2], stream[0]);

            double nor;
            nor=lastwave.normalize(stream[0]);
            beta.push_back(nor);
            myv[(inner+1)%2].set(lastwave,stream[0]);
            inner++;
            icount++;
        }
        

        wave_store[0].mul_num(vec[0]);
        for (size_t i=1; i<inner-1; ++i) {
            wave_store[0].mul_add(vec[i], wave_store[i]);
        }
        wave_store[0].toGPU(lastwave, stream[0]);
        
        double nor;
        nor=lastwave.normalize(stream[0]);

        cout << "memCPU = " << memsize/1024 <<  "GB, outter iter " << iout+1 << " end, wave_norm = " << nor << endl;

        vector<thread> th;
        for (size_t i = 1; i<inner-1; i++) {
            th.push_back(thread(clear_help, ref(wave_store[i])));
        }
        for (size_t i = 0; i < 2; i++) {
            myv[i].clear();
        }
        for(auto &t : th){
            t.join();
        }
        wave_store[0].clear();
        delete [] wave_store; wave_store=NULL;
        iout++;
    }
    
    cublasSetStream(GlobalHandle, 0);
    for (size_t i = 0; i < 2; i++) {
        cudaStreamDestroy(stream[i]);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    time_2+=duration.count()/1000.0;
    cout << "Energy=" << energy/ltot << ", iter=" << icount << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
    string name="out/energy.dat";
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific << "Energy=" << energy << ", iter=" << icount << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
        out.close();
    }
}

void lanc_main_V2(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis) {
    auto start = high_resolution_clock::now();
    cudaStream_t stream[2];
    for (size_t j = 0; j < 2; j++) {
        cudaStreamCreate(&stream[j]);
    }
    
    int time2=0;
    cublasSetStream(GlobalHandle, stream[0]);
    // wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
    wave_CPU wave_store;
    vector<double> alpha, beta;
    vector<double> vec;
    wave myv[2];
    myv[0]=lastwave;
    wave_store.construct(lastwave);
    double err=1.0, energy=1000.0;
    int iout=0, icount=0;
    double Hmem=max(sys.mem_size(),env.mem_size());
    cout << endl;
    cout << "sysmem= " << sys.mem_size()/1024 << "GB, envmem= " << env.mem_size()/1024 << "GB, 3*wavmem= " << 3*lastwave.mem_size()/1024 << "GB" << endl;
    while ( iout<Lan_max_outter && err>lan_error) {
        cout << "outter iter: " << iout+1 << endl;
        alpha.clear(); beta.clear();
        beta.push_back(0); alpha.push_back(0);//skip [0]
        beta.push_back(0);// set b[1]
        myv[0].setzero(stream[0]);
        myv[1].set(lastwave,stream[0]);
        wave_store.copy(lastwave, stream[1]);
        double memsize=0;
        int inner=1;
        cout << "inner iter: ";
        while ( inner < 4 ) {
            Htowave(sys, myv[inner%2], lastwave, env, sys_basis, env_basis, stream[0]);
            alpha.push_back( lastwave.dot(myv[inner%2]) ); // a(i)
            cout << inner << ": a=" << alpha[inner] << endl;
            cout << inner << ": b=" << beta[inner] << endl;
            if (inner>1) {
                double* diag = new double[inner]();
                double* ndiag= new double[inner]();
                double* dwork= new double[inner*inner]();
                for (size_t j = 1; j < inner+1; ++j) {
                    diag[j-1]=alpha[j];
                }
                for (size_t j = 2; j < inner+1; ++j) {
                    ndiag[j-2]=beta[j];
                }
                LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', inner, diag, ndiag, dwork, inner);
                err=energy-diag[0];
                cout << " err=" << err;
                energy=diag[0];
                delete [] diag; diag=NULL;
                delete [] ndiag; ndiag=NULL;
                vec.clear();
                for (size_t j = 0; j < inner; ++j) {
                    vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
                }
                delete [] dwork; dwork=NULL;
            }
            cout << "; ";
            if (inner<3) {
                lastwave.mul_add(-alpha[inner], myv[inner%2], stream[0]);
                lastwave.mul_add(-beta[inner], myv[(inner-1)%2], stream[0]);

                double nor;
                nor=lastwave.normalize(stream[0]);
                beta.push_back(nor);
                myv[(inner+1)%2].set(lastwave,stream[0]);
            }
            inner++;
        }
        // wave_store, myv[0], myv[1]

        wave_store.mul_num(vec[0]);
        wave_store.toGPU(lastwave, stream[0]);
        lastwave.mul_add(vec[1],myv[0],stream[0]);
        lastwave.mul_add(vec[2],myv[1],stream[0]);
        cout << endl;
        iout++;
    }
    
    cublasSetStream(GlobalHandle, 0);
    for (size_t i = 0; i < 2; i++) {
        cudaStreamDestroy(stream[i]);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    time_2+=duration.count()/1000.0;
    cout << "Energy=" << energy/ltot << ", iter=" << 3*iout << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
    string name="out/energy.dat";
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific << "Energy=" << energy << ", iter=" << 3*iout << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
        out.close();
    }
}

void lanc_main_V3(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, wave_CPU (&wave_store)[2], cudaStream_t stream[2]) {
    auto start = high_resolution_clock::now();
    
    int time2=0;
    // wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
    vector<double> alpha, beta;
    vector<double> vec;
    wave myv;
    double err=1.0, energy=1000.0;
    int iout=0, icount=0;
    double Hmem=max(sys.mem_size(),env.mem_size());
    cout << endl;
    cout << "sysmem= " << sys.mem_size()/1024 << "GB, envmem= " << env.mem_size()/1024 << "GB, 2*wavmem= " << 2*lastwave.mem_size()/1024 << "GB" << endl;
    myv.set(lastwave,stream[0]);
    while ( iout<Lan_max_outter && err>lan_error) {
        cout << "outter iter: " << iout+1 << endl;
        alpha.clear(); beta.clear();
        beta.push_back(0); alpha.push_back(0);//skip [0]
        beta.push_back(0);// set b[1]
        cudaDeviceSynchronize();
        wave_store[0].copy(myv, stream[1]);
        double memsize=0;
        int inner=1;
        cout << "inner iter: ";
        while ( inner < 4 ) {
            Htowave(sys, myv, lastwave, env, sys_basis, env_basis, stream[0]);
            alpha.push_back( lastwave.dot(myv) ); // a(i)
            // exit(3);
            if (inner>1) {
                double* diag = new double[inner]();
                double* ndiag= new double[inner]();
                double* dwork= new double[inner*inner]();
                for (size_t j = 1; j < inner+1; ++j) {
                    diag[j-1]=alpha[j];
                }
                for (size_t j = 2; j < inner+1; ++j) {
                    ndiag[j-2]=beta[j];
                }
                LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', inner, diag, ndiag, dwork, inner);
                err=energy-diag[0];
                cout << " err=" << err;
                energy=diag[0];
                delete [] diag; diag=NULL;
                delete [] ndiag; ndiag=NULL;
                vec.clear();
                for (size_t j = 0; j < inner; ++j) {
                    vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
                }
                delete [] dwork; dwork=NULL;
            }
            cout << "; ";
            if (inner<3) {
                lastwave.mul_add(-alpha[inner], myv, stream[0]);
                if (inner>1) {
                    cudaDeviceSynchronize();
                    wave_store[inner-2].toGPU(myv, stream[0]);// need opt
                    lastwave.mul_add(-beta[inner], myv, stream[0]);
                }
                double nor;
                nor=lastwave.normalize(stream[0]);
                beta.push_back(nor);
                myv.set(lastwave,stream[0]);
                if (inner<2) {
                    wave_store[inner].copy(myv, stream[1]);
                }
            }
            inner++;
        }
        // wave_store[0], wave_store[1], myv
        myv.num_mul(vec[2],stream[1]);

        wave_store[0].mul_num(vec[0]);
        wave_store[0].mul_add(vec[1], wave_store[1]);
        wave_store[0].toGPU(lastwave, stream[0]);// need opt
        // lastwave.mul_add(vec[2],myv,stream[0]);

        cudaDeviceSynchronize();
        myv.mul_add(1,lastwave,stream[0]);
        
        cout << endl;
        iout++;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    time_2+=duration.count()/1000.0;
    cout << "Energy=" << energy/ltot << ", iter=" << 3*iout << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
    string name="out/energy.dat";
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific << "Energy=" << energy << ", iter=" << 3*iout << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
        out.close();
    }
}

void lanc_main_V4(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, wave_CPU (&wave_store)[4], cudaStream_t stream[2]) {
    auto start = high_resolution_clock::now();
    
    int time2=0;
    // wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
    vector<double> alpha, beta;
    vector<double> vec;
    double err=1.0, energy=1000.0;
    int iout=0, icount=0;
    double Hmem=max(sys.mem_size(),env.mem_size());
    cout << endl;
    cout << "sysmem= " << sys.mem_size()/1024 << "GB, envmem= " << env.mem_size()/1024 << "GB, wavmem= " << lastwave.mem_size()/1024 << "GB" << endl;
    while ( iout<Lan_max_outter && err>lan_error) {
        cout << "outter iter: " << iout+1 << endl;
        alpha.clear(); beta.clear();
        beta.push_back(0); alpha.push_back(0);//skip [0]
        beta.push_back(0);// set b[1]
        wave_store[0].copy(lastwave, stream[1]);
        double memsize=0;
        int inner=1;
        cout << "inner iter: ";
        while ( inner < 4 ) {
            wave_store[inner].setzero();
            HtowaveGtoC(sys, lastwave, wave_store[inner], env, sys_basis, env_basis, stream[0]);// to do
            alpha.push_back( wave_store[inner].dot(lastwave, stream[0]) ); // a(i)
            cout << "alpha=" << alpha[inner] << endl;
            // exit(3);
            if (inner>1) {
                double* diag = new double[inner]();
                double* ndiag= new double[inner]();
                double* dwork= new double[inner*inner]();
                for (size_t j = 1; j < inner+1; ++j) {
                    diag[j-1]=alpha[j];
                }
                for (size_t j = 2; j < inner+1; ++j) {
                    ndiag[j-2]=beta[j];
                }
                LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', inner, diag, ndiag, dwork, inner);
                err=energy-diag[0];
                cout << " err=" << err;
                energy=diag[0];
                delete [] diag; diag=NULL;
                delete [] ndiag; ndiag=NULL;
                vec.clear();
                for (size_t j = 0; j < inner; ++j) {
                    vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
                }
                delete [] dwork; dwork=NULL;
            }
            cout << "; ";
            if (inner<3) {
                wave_store[inner].mul_add(-alpha[inner], wave_store[inner-1]);
                if (inner>1) {
                    wave_store[inner].mul_add(-beta[inner], wave_store[inner-2]);
                }
                double nor;
                nor=wave_store[inner].normalize();
                beta.push_back(nor);
                wave_store[inner].toGPU(lastwave, stream[0]);
            }
            inner++;
        }
        // wave_store[0], wave_store[1], wave_store[2]
        wave_store[0].mul_num(vec[0]);
        wave_store[0].mul_add(vec[1], wave_store[1]);
        wave_store[0].mul_add(vec[2], wave_store[2]);
        wave_store[0].toGPU(lastwave, stream[0]);// need opt
        cudaDeviceSynchronize();
        cout << endl;
        iout++;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    time_2+=duration.count()/1000.0;
    cout << "Energy=" << energy/ltot << ", iter=" << 3*iout << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
    string name="out/energy.dat";
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific << "Energy=" << energy << ", iter=" << 3*iout << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
        out.close();
    }
}


// wave lanc_main_new(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis) {
//     auto start = high_resolution_clock::now();
//     cudaStream_t stream[2];
//     for (size_t j = 0; j < 2; j++) {
//         cudaStreamCreate(&stream[j]);
//     }
    
//     int time2=0;
//     cublasSetStream(GlobalHandle, stream[0]);
//     // wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
//     vector<double> alpha, beta;
//     vector<double> vec;
//     wave newwave(lastwave, 0);
//     wave myv[2];
//     myv[0]=lastwave;
//     myv[1]=lastwave;
//     double err=1.0, energy=1000.0;
//     int iout=0, icount=0;
//     double Hmem=max(sys.mem_size(),env.mem_size());
//     cout << endl;
//     cout << "Hmem= " << Hmem/1024 << "GB" << endl;
//     while ( iout<Lan_max_outter && err>lan_error) {
//         cout << "outter iter: " << iout+1 << endl;
//          // be careful about Lan_max 
//         alpha.clear(); beta.clear();
//         beta.push_back(0); alpha.push_back(0);//skip [0]
//         beta.push_back(0);// set b[1]
//         myv[0].setzero(stream[0]);
//         myv[1].set(newwave,stream[0]);
//         double memsize=0;
//         int upperbound=100;
//         int inner=1;
//         cout << "inner iter: ";
//         wave_CPU* wave_store = new wave_CPU[Lan_max_inner+1]();
//         while ( inner < min(upperbound, Lan_max_inner) && err>lan_error ) {
//             cout << inner;
        
//             auto start2 = high_resolution_clock::now();
//             thread th0(store_help, ref(newwave), ref(wave_store[inner-1]));
//             auto stop2 = high_resolution_clock::now();
//             auto duration2 = duration_cast<milliseconds>(stop2 - start2);
//             time2+=duration2.count();
//             newwave.set(Htowave(sys, myv[inner%2], env, sys_basis, env_basis, stream[0]), stream[0]);
//             alpha.push_back( newwave.dot(myv[inner%2]) ); // a(i)
//             if (inner>1) {
//                 double* diag = new double[inner]();
//                 double* ndiag= new double[inner]();
//                 double* dwork= new double[inner*inner]();
//                 for (size_t j = 1; j < inner+1; ++j) {
//                     diag[j-1]=alpha[j];
//                 }
//                 for (size_t j = 2; j < inner+1; ++j) {
//                     ndiag[j-2]=beta[j];
//                 }
//                 LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', inner, diag, ndiag, dwork, inner);
//                 err=energy-diag[0];
//                 cout << ", err=" << err;
//                 energy=diag[0];
//                 delete [] diag; diag=NULL;
//                 delete [] ndiag; ndiag=NULL;
//                 vec.clear();
//                 for (size_t j = 0; j < inner; ++j) {
//                     vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
//                 }
//                 delete [] dwork; dwork=NULL;
//             }
//             cout << "; " << endl;
//             th0.join();
//             if (inner==1) {upperbound=int((33000-Hmem)/wave_store[0].mem_size());}
//             wave_store[inner-1].copy(myv[inner%2], stream[1]);
//             memsize+=wave_store[inner-1].mem_size();

//             newwave.mul_add(-alpha[inner], myv[inner%2], stream[0]);
//             newwave.mul_add(-beta[inner], myv[(inner-1)%2], stream[0]);

//             double nor;
//             nor=newwave.normalize(stream[0]);
//             beta.push_back(nor);
//             myv[(inner+1)%2].set(newwave,stream[0]);
//             inner++;
//             icount++;
//         }
//         cout << "memCPU = " << memsize/1024 <<  "GB, outter iter " << iout+1 << " end" << endl;
//         wave_store[0].mul_num(vec[0]);
//         for (size_t i=1; i<inner-1; ++i) {
//             wave_store[0].mul_add(vec[i], wave_store[i]);
//         }
//         wave_store[0].toGPU(newwave, stream[0]);
        
//         vector<thread> th;
//         for (size_t i = 1; i<inner-1; i++) {
//             th.push_back(thread(clear_help, ref(wave_store[i])));
//         }
//         for (size_t i = 0; i < 2; i++) {
//             myv[i].clear();
//         }
//         for(auto &t : th){
//             t.join();
//         }
//         wave_store[0].clear();
//         delete [] wave_store; wave_store=NULL;
//         iout++;
//     }
//     // vector<thread> th;
//     // for (size_t i = 1; i < alpha.size()-1; i++) {
//     //     th.push_back(thread(clear_help, ref(wave_store[i])));
//     // }
//     // for (size_t i = 0; i < 2; i++) {
//     //     myv[i].clear();
//     // }
//     // for(auto &t : th){
//     //     t.join();
//     // }
//     // wave_store[0].clear();
//     // delete [] wave_store; wave_store=NULL;
    
//     cublasSetStream(GlobalHandle, 0);
//     for (size_t i = 0; i < 2; i++) {
//         cudaStreamDestroy(stream[i]);
//     }


//     auto stop = high_resolution_clock::now();
//     auto duration = duration_cast<milliseconds>(stop - start);
//     time_2+=duration.count()/1000.0;
//     cout << "Energy=" << energy/ltot << ", iter=" << icount << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
//     string name="out/energy.dat";
//     ofstream out(name.c_str(), ios::out | ios::app);
//     if (out.is_open()) {
//         out << scientific << "Energy=" << energy << ", iter=" << icount << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
//         out.close();
//     }
//     return newwave;
// }

// wave lanc_main_new(const Hamilton &sys, wave &lastwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis) {
//     auto start = high_resolution_clock::now();
//     wave_CPU* wave_store = new wave_CPU[Lan_max+1]();
//     wave myv[2];
//     vector<double> alpha, beta;
//     wave newwave(lastwave, 0);
//     newwave.setzero(0);
//     myv[0]=newwave;
//     beta.push_back(0); alpha.push_back(0);//skip [0]
//     beta.push_back(0);// set b[1]
//     myv[1]=lastwave;

//     double err=1.0, energy=1000.0;
// 	vector<double> vec;
//     int time2=0;
//     cudaStream_t stream[2];
//     for (size_t j = 0; j < 2; j++) {
//         cudaStreamCreate(&stream[j]);
//     }

//     double memsize=0;
//     cout << endl;
//     int upperbound=100;
//     int i=1;
//     while ( i < upperbound && err>lan_error ) {
//         cout << "iter: " << i;
//         auto start2 = high_resolution_clock::now();
//         thread th0(store_help, ref(newwave), ref(wave_store[i-1]));
        
//         // wave_store[i-1].fromGPU(myv[i%3], stream[1]);
//         // wave_store[i-1].construct(lastwave);
        
//         auto stop2 = high_resolution_clock::now();
//         auto duration2 = duration_cast<milliseconds>(stop2 - start2);
//         time2+=duration2.count();
//         wave tmp;
//         cublasSetStream(GlobalHandle, stream[0]);
//         tmp.set(Htowave(sys, myv[i%2], env, sys_basis, env_basis, stream[0]), stream[0]);

//         alpha.push_back( tmp.dot(myv[i%2]) ); // a(i)
//         if (i>2) {
// 			double* diag = new double[alpha.size()-1]();
// 			double* ndiag= new double[alpha.size()-1]();
// 			double* dwork= new double[(alpha.size()-1)*(alpha.size()-1)]();
// 			for (size_t j = 1; j < alpha.size(); ++j) {
// 				diag[j-1]=alpha[j];
// 			}
// 			for (size_t j = 2; j < alpha.size(); ++j) {
// 				ndiag[j-2]=beta[j];
// 			}
// 			LAPACKE_dstev (LAPACK_COL_MAJOR, 'V', alpha.size()-1, diag, ndiag, dwork, alpha.size()-1);
//             err=energy-diag[0];
//             energy=diag[0];
// 			delete [] diag; diag=NULL;
// 			delete [] ndiag; ndiag=NULL;
// 			vec.clear();
// 			for (size_t j = 0; j < alpha.size()-1; ++j) {
// 				vec.push_back(dwork[ j ]);//*(alpha.size()-1) ]);
// 			}
// 			delete [] dwork; dwork=NULL;
//         }
//         th0.join();
//         upperbound=int(36000/wave_store[0].mem_size())+1;
//         wave_store[i-1].copy(myv[i%2], stream[1]);
//         memsize+=wave_store[i-1].mem_size();
//         cout << ", transfered " << memsize/1024 << "GB";

//         tmp.mul_add(-alpha[i], myv[i%2], stream[0]);
//         tmp.mul_add(-beta[i], myv[(i-1)%2], stream[0]);

//         double nor;
//         nor=tmp.normalize(stream[0]);
//         beta.push_back(nor);
//         myv[(i+1)%2].set(tmp,stream[0]);
//         i++;
//         tmp.clear();
//     }
//     cout << "iter: end" << endl;
//     for (size_t i = 0; i < 2; i++) {
//         myv[i].clear();
//     }
//     // delete [] myv; myv=NULL;

//     auto start1 = high_resolution_clock::now();
//     wave_store[0].mul_num(vec[0]);
//     for (size_t i=1; i<alpha.size()-1; ++i) {
//         wave_store[0].mul_add(vec[i], wave_store[i]);
//     }
//     cout << "wave added" << endl;
//     // for (size_t i = 1; i < int(alpha.size()/2); i++) {
//     //     if (2*i-1<alpha.size()-1) {
//     //         thread th0(store_help, ref(newwave), ref(wave_store[2*i-1]));
//     //         if (2*i<alpha.size()-1) {wave_store[2*i].clear();}
//     //         th0.join();
//     //     }
//     // }
//     vector<thread> th;
//     for (size_t i = 1; i < alpha.size()-1; i++) {
//         th.push_back(thread(clear_help, ref(wave_store[i])));
//     }
//     wave_store[0].toGPU(newwave, stream[0]);
//     for(auto &t : th){
//         t.join();
//     }
//     wave_store[0].clear();
//     delete [] wave_store; wave_store=NULL;
//     cout << "wave_store released" << endl;
//     // wave tmp;
//     // wave_store[0].constructGPU(tmp);
//     // for (size_t i=0; i<alpha.size()-1; ++i) {
//     //     // tmp.fromC(wave_store[i], stream[0]);
//     //     wave_store[i].copyGPU(tmp, stream[0]);
//     //     cudaDeviceSynchronize();
//     //     newwave.mul_add(vec[i], tmp, stream[1]);
//     //     // newwave.mul_add(vec[i], myv[i+1]);
//     // }

//     cublasSetStream(GlobalHandle, 0);
//     for (size_t i = 0; i < 2; i++) {
//         cudaStreamDestroy(stream[i]);
//     }



//     auto stop = high_resolution_clock::now();
//     auto duration = duration_cast<milliseconds>(stop - start);
//     auto duration1 = duration_cast<milliseconds>(stop - start1);
//     time_2+=duration.count()/1000.0;
//     cout << "Energy=" << energy/ltot << ", iter=" << i-1 << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
//     cout << "toC=" << time2 << "ms" << ", fromC=" << duration1.count() << "ms" << endl;
//     string name="out/energy.dat";
//     ofstream out(name.c_str(), ios::out | ios::app);
//     if (out.is_open()) {
//         out << scientific << "Energy=" << energy << ", iter=" << i-1 << ", lan_err="<< err << ", lanc time=" << duration.count() << "ms" << endl;
//         out << "toC=" << time2 << "ms" << ", fromC=" << duration1.count() << "ms" << endl;
//         out.close();
//     }
//     return newwave;
// }
