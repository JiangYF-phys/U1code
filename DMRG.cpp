#include "DMRG.hpp"
//using namespace chrono;

extern wave ground;

void nontrunc() {
    makeJsite().optodisk(file+"sysbl"+to_string(1));
    site.optodisk(file+"envbl"+to_string(1));
    
    for (int i=1; i<truncpoint; ++i) {
        Hamilton sys(file+"sysbl"+to_string(i)), env(file+"envbl"+to_string(i));
        vector<repmap> sysbasis, envbasis;
        sysbasis=jmap(sys.Ham,site.Ham);
        envbasis=jmap(site.Ham,env.Ham);
        maptodisk(sysbasis, file+"sysmap"+to_string(i+1));
        maptodisk(envbasis, file+"envmap"+to_string(i+1));
        sys=addonesite(sys, sysbasis, 's');
        sys.optodisk(file+"sysbl"+to_string(i+1));
        env=addonesite(env, envbasis, 'e');
        env.optodisk(file+"envbl"+to_string(i+1));
        // cout << sys.Ham << endl;
    }
}

/*
 :            sys         env
 l         1 2 3 4 5 | 6 7 8 9 10
 site      x x x x . | . x x x x
 i   mysys 0 1 2 3 4 | 4 3 2 1 0 myenv
 */
void warmup() {
    
	nontrunc();

    for (int i=truncpoint; i<middleid; ++i) {
        cout << "syslen=" << i+1 << endl;
		string name="out/energy.dat";
		ofstream out(name.c_str(), ios::out | ios::app);
		if (out.is_open()) {
			out << "syslen=" << i+1 << endl;
			out.close();
		}
        Hamilton sys(file+"sysbl"+to_string(i)), env(file+"envbl"+to_string(i));
        vector<repmap> sysbasis, envbasis;
        sysbasis=jmap(sys.Ham,site.Ham);
        envbasis=jmap(site.Ham,env.Ham);
        maptodisk(sysbasis, file+"sysmap"+to_string(i+1));
        maptodisk(envbasis, file+"envmap"+to_string(i+1));
        
        wave myw;
        
		// int jtot0=max(0,int(2*jtot*(1-(i+1)*2.0/ltot)));
        int jtot0=int(jtot*(i+1)*2.0/ltot+0.001);
		int ntot0=int(ntot*(i+1)*2.0/ltot+0.001);
		if (adjustflag==1 && jtot0%2 != ntot0%2) {
			cout << "jtot0=" << jtot0 << " and ntot0=" << ntot0 << " don't match, try increasing 2*J by 1" << endl;
			jtot0++;
		}

        myw.initial(sysbasis, envbasis, jtot0, ntot0);
        myw.setran(); myw.normalize(0);
        lanc_main_new(sys, myw, env, sysbasis, envbasis);

        // wave_CPU wave_store[2];
        // wave_store[0].construct(myw);
        // wave_store[1].construct(myw);
        // cublasSetStream(GlobalHandle, 0);
        // lanc_main_V3(sys, myw, env, sysbasis, envbasis, wave_store, 0);

        reducematrix rou, trunc;
        rou=wavetorou(myw, 's', 0);
        trunc=routotrunc(rou, false);
		trunc.todisk(file+"systrun"+to_string(i+1), 'n');
        sys=addonesite(sys, sysbasis, 's');
		sys.truncHam(trunc);
		sys.optodisk(file+"sysbl"+to_string(i+1));
        
        cout << endl;
        name="out/energy.dat";
        out.open(name.c_str(), ios::out | ios::app);
        if (out.is_open()) {
            out << endl;
            out.close();
        }
        
        rou.clear();trunc.clear();
        rou=wavetorou(myw, 'e', 0);
        trunc=routotrunc(rou, false);
		trunc.todisk(file+"envtrun"+to_string(i+1), 'n');
        env=addonesite(env, envbasis, 'e');
		env.truncHam(trunc);
		env.optodisk(file+"envbl"+to_string(i+1));
        
        cout << endl;
        name="out/energy.dat";
        out.open(name.c_str(), ios::out | ios::app);
        if (out.is_open()) {
            out << endl;
            out.close();
        }
    }
}

void reconstruct() {
    cout << "CPUmem_0="<< physical_memory_used_by_process() << "GB" << endl;
    nontrunc();
    cout << "CPUmem_1="<< physical_memory_used_by_process() << "GB" << endl;
    for (int i=truncpoint; i<beginid; ++i) {
        Hamilton sys(file+"sysbl"+to_string(i));
        vector<repmap> sysbasis=jmap(sys.Ham,site.Ham);
        maptodisk(sysbasis, file+"sysmap"+to_string(i+1));
        reducematrix trunc;
        trunc.fromdisk(file+"systrun"+to_string(i+1), 'n', 0);
        addonesite_replace(sys, sysbasis, trunc, 's');
		sys.optodisk(file+"sysbl"+to_string(i+1));
        // std::cout << sys.Ham << std::endl;
    }
    for (int i=truncpoint; i<ltot-beginid-2; ++i) {
        Hamilton env(file+"envbl"+to_string(i));
        vector<repmap> envbasis=jmap(site.Ham,env.Ham);
        maptodisk(envbasis, file+"envmap"+to_string(i+1));
        reducematrix trunc;
        trunc.fromdisk(file+"envtrun"+to_string(i+1), 'n', 0);
        addonesite_replace(env, envbasis, trunc, 'e');
		env.optodisk(file+"envbl"+to_string(i+1));
    }
    cout << "CPUmem_2="<< physical_memory_used_by_process() << "GB" << endl;
}

void flipwave() {
    vector<repmap> sysbasis, envbasis;
    mapfromdisk(sysbasis, file+"sysmap"+to_string(beginid));
    mapfromdisk(envbasis, file+"envmap"+to_string(ltot-beginid));
    wave myw;
    myw.initial(sysbasis, envbasis, jtot, ntot);
    myw.flipsyssite(sysbasis, ground); 
    double nor;
    nor=myw.normalize(0);
    ground=myw;
    cout << "spin flipped at " << beginid << " point, norm:" << nor << endl;
}

void savesyshelp(const Hamilton &tot, const reducematrix &trun, const int &site) {
    if (site>0) {
        trun.todisk(file+"systrun"+to_string(site), 'n');
        tot.optodisk(file+"sysbl"+to_string(site));
    }
} 

void readenvhelp(HamiltonCPU &tot, const int &site) {
    if (site>0) {
        tot.fromdisk(file+"envbl"+to_string(site));
    }
    // thread::id tID= this_thread::get_id();
    // cout << endl << " read env thread " << tID << ": job done. " << endl;
} 
/*
 :            sys         env
 l         1 2 3 4 5 | 6 7 8 9 10
site       x x x x . | . x x x x
 i   mysys 0 1 2 3 4 | 4 3 2 1 0 myenv
 */
//                   =5              =10-truncpoint-1
void LtoR(const int &beg, const int &end, const bool &initial, const bool &continu) {
    time_1=0;time_2=0;time_3=0;time_4=0;
    cudaStream_t stream[2];
    for (size_t j = 0; j < 2; j++) {
        cudaStreamCreate(&stream[j]);
    }
    cout << "CPUmem_0="<< physical_memory_used_by_process() << "GB" << endl;
    Hamilton sys(file+"sysbl"+to_string(beg-1));
    HamiltonCPU env_pre(file+"envbl"+to_string(ltot-beg-1));
    cout << "CPUmem_1="<< physical_memory_used_by_process() << "GB" << endl;
    Hamilton env;
    reducematrix systrun;
    if (continu) systrun.fromdisk(file+"systrun"+to_string(beg-1), 'n', 0);
    for (int i=beg-1; i<end; ++i) {    //i=4
        auto start = high_resolution_clock::now();
        cout << "syslen=" << i+1 << endl;
        string name="out/energy.dat";
        ofstream out(name.c_str(), ios::out | ios::app);
        if (out.is_open()) {
            out << endl << "syslen=" << i+1 << endl;
            out.close();
        }
        env.fromC(env_pre);
        cout << "CPUmem_a="<< physical_memory_used_by_process() << "GB" << endl;
        env_pre.clear();
        cout << "CPUmem_b="<< physical_memory_used_by_process() << "GB" << endl;

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "prep time1=" << duration.count() << "ms, ";
        start = high_resolution_clock::now();

        int site_e=0;
        if (i<end-1) {
            site_e=ltot-i-1-2;
            // improve: allocate than read, parallel read
            env_pre.fromdisk(file+"envbl"+to_string(site_e));
        }
        // thread th0(readenvhelp, ref(env_pre), ref(site_e) );
        
        int site_s=0;
        if (i>beg-1) {site_s=i;}
        thread th1(savesyshelp, ref(sys), ref(systrun), ref(site_s) );

        vector<repmap> sysbasis, envbasis;
        sysbasis=jmap(sys.Ham,site.Ham);

        if (i==beg-1 && !continu) {
            envbasis=jmap(site.Ham,env.Ham);
            if (initial) {
                wave myw;
                myw.initial(sysbasis, envbasis, jtot, ntot);
                myw.setran();
                ground=myw;
            }
        } else {
            mapfromdisk(envbasis, file+"envmap"+to_string(ltot-i-1));
            wave myw;
            reducematrix envtrun;
            envtrun.fromdisk(file+"envtrun"+to_string(ltot-i-1), 'n', 0);
            vector<repmap> envbs;
            mapfromdisk(envbs, file+"envmap"+to_string(ltot-i));
            myw.initial(sysbasis, envbasis, jtot, ntot);
            myw.transLtoR(ground, systrun, envtrun, sysbasis, envbs);
            ground=myw;
        }
        // construct groundCPU
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << "prep time2=" << duration.count() << "ms, ";
        time_1+=duration.count()/1000.0;

        cout << "trunmem=" << systrun.mem_size()/1024 << "GB, "; 
        ground.normalize(0);
        cublasSetStream(GlobalHandle, stream[0]);

        cout << endl << "CPUmem_0="<< physical_memory_used_by_process() << "GB";

        wave_CPU wave_store[2];

        if (lanczos_ver==1) {
            lanc_main_new(sys, ground, env, sysbasis, envbasis);
            wave_store[0].construct(ground);
        } else if (lanczos_ver==3) {
            for (size_t j = 0; j < 2; j++) {wave_store[j].construct(ground);}
            lanc_main_V3(sys, ground, env, sysbasis, envbasis, wave_store, stream);
            wave_store[1].clear();
        }

        cout << "CPUmem_1="<< physical_memory_used_by_process() << "GB" << endl;
        // wave_CPU wave_store[4];
        // for (size_t j = 0; j < 4; j++) {wave_store[j].construct(ground);}
        // lanc_main_V4(sys, ground, env, sysbasis, envbasis, wave_store, stream);
        
        
        env.clear();
        // save ground to CPU
        wave_store[0].copy(ground, stream[1]);
        start = high_resolution_clock::now();
        reducematrix rou;
        rou.set(wavetorou(ground, 's', stream[0]),stream[0]);
        cout << "GPUmem_1="<< GPU_memory_used_by_process() << "GB" << endl;
        th1.join();
        cublasSetStream(GlobalHandle, 0);
        systrun=routotrunc(rou, i+1==lx*ly/2);

        rou.clear();
        cudaDeviceSynchronize();
        ground.clear();                                                                                                                                                                                                                              
        cout << endl << "sysmem=" << sys.mem_size()/1024 << "GB, systrunmem=" << systrun.mem_size()/1024 << "GB ";
        
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << endl << "trun time=" << duration.count() << "ms";
        time_3+=duration.count()/1000.0;

        start = high_resolution_clock::now();
        addonesite_replace(sys, sysbasis, systrun, 's');

        // read ground back
        wave_store[0].toGPU(ground, 0);
        
        maptodisk(sysbasis, file+"sysmap"+to_string(i+1));
        if (i<end-1) {
            remove((file+"envbl"+to_string(ltot-i-2)).c_str());
            remove((file+"envtrun"+to_string(ltot-i-1)).c_str());
        }
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << ", newH time=" << duration.count() << "ms";
        time_4+=duration.count()/1000.0;

        start = high_resolution_clock::now();
        // th0.join();
        cout << ", env memCPU=" << env_pre.mem_size()/1024 << "GB";

        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << ", res time=" << duration.count() << "ms" << endl;

        cout << "CPUmem_2="<< physical_memory_used_by_process() << "GB" << endl;
    }
    
    systrun.todisk(file+"systrun"+to_string(end), 'n');
    sys.optodisk(file+"sysbl"+to_string(end));
    cout << "prep time=" << time_1 << "s" << endl;
    cout << "lanc time=" << time_2 << "s" << endl;
    cout << "trun time=" << time_3 << "s" << endl;
    cout << "newH time=" << time_4 << "s" << endl;
    for (size_t i = 0; i < 2; i++) {
        cudaStreamDestroy(stream[i]);
    }
}

void saveenvhelp(const Hamilton &tot, const reducematrix &trun, const int &site) {
    if (site>0) {
        trun.todisk(file+"envtrun"+to_string(site), 'n');
        tot.optodisk(file+"envbl"+to_string(site));
    }
} 

void readsyshelp(HamiltonCPU &tot, const int &site) {
    if (site>0) {
        tot.fromdisk(file+"sysbl"+to_string(site));
    }
    // thread::id tID= this_thread::get_id();
    // cout << endl << " read sys thread " << tID << ": job done. " << endl;
}

//                   =10-tp-1          =tp+1
void RtoL(const int &beg, const int &end, const bool &initial) {
    time_1=0;time_2=0;time_3=0;time_4=0;
    cudaStream_t stream[2];
    for (size_t j = 0; j < 2; j++) {
        cudaStreamCreate(&stream[j]);
    }
	Hamilton env(file+"envbl"+to_string(ltot-beg-1));
    HamiltonCPU sys_pre(file+"sysbl"+to_string(beg-1));
    Hamilton sys;
    reducematrix envtrun;
    string m_name="out/n.dat";
    ofstream mout(m_name.c_str(), ios::out | ios::app);
    mout << scientific;
    for (int i=ltot-beg-1; i<ltot-end; ++i) {    //i=tp
        auto start = high_resolution_clock::now();
        cout << "syslen=" << ltot-i-1 << endl;
        string name="out/energy.dat";
        ofstream out(name.c_str(), ios::out | ios::app);
        if (out.is_open()) {
            out << endl << "syslen=" << ltot-i-1 << endl;
            out.close();
        }
        
        sys.fromC(sys_pre);
        sys_pre.clear();

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "prep time1=" << duration.count() << "ms, ";
        start = high_resolution_clock::now();

        int site_s=0;
        if (i<ltot-end-1) {
            site_s=ltot-i-1-2;
            sys_pre.fromdisk(file+"sysbl"+to_string(site_s));
        }
        // thread th0(readsyshelp, ref(sys_pre), ref(site_s) );
        int site_e=0;
        if (i>ltot-beg-1) {site_e=i;}
        thread th1(saveenvhelp, ref(env), ref(envtrun), ref(site_e) );

        vector<repmap> sysbasis, envbasis;
        envbasis=jmap(site.Ham,env.Ham);
        
        if (i==ltot-beg-1) {
            sysbasis=jmap(sys.Ham,site.Ham);
            if (initial) {
                wave myw;
                myw.initial(sysbasis, envbasis, jtot, ntot);
                myw.setran();
                ground=myw;
            }
        } else {
            mapfromdisk(sysbasis, file+"sysmap"+to_string(ltot-i-1));
            wave myw;
            reducematrix systrun;
            systrun.fromdisk(file+"systrun"+to_string(ltot-i-1), 'n', 0);
            vector<repmap> sysbs;
            mapfromdisk(sysbs, file+"sysmap"+to_string(ltot-i));
            myw.initial(sysbasis, envbasis, jtot, ntot);
            myw.transRtoL(ground, systrun, envtrun, sysbs, envbasis);
            ground=myw;
        }

        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << "prep time2=" << duration.count() << "ms, ";
        time_1+=duration.count()/1000.0;

        cout << "trunmem=" << envtrun.mem_size()/1024 << "GB, "; 
        ground.normalize(0);
        cublasSetStream(GlobalHandle, stream[0]);
        wave_CPU wave_store[2];

        if (lanczos_ver==1) {
            lanc_main_new(sys, ground, env, sysbasis, envbasis);
            wave_store[0].construct(ground);
        } else if (lanczos_ver==3) {
            // wave myw;
            // myw=ground;
            // lanc_main_new(sys, myw, env, sysbasis, envbasis);

            for (size_t j = 0; j < 2; j++) {wave_store[j].construct(ground);}
            lanc_main_V3(sys, ground, env, sysbasis, envbasis, wave_store, stream);
            wave_store[1].clear();
        }
        // wave_CPU wave_store[4];
        // for (size_t j = 0; j < 4; j++) {wave_store[j].construct(ground);}
        // lanc_main_V4(sys, ground, env, sysbasis, envbasis, wave_store, stream);

        if (mout.is_open()) {
            wave myw;
            myw=ground;
            if (i==ltot-beg-1) {
                myw.setzero(stream[0]);
                myw.mul(sys.Ham, siteCPU.Ham, ground, fn_CPU, env.Ham, 1.0, "iini", sysbasis, envbasis, stream[0]);
                mout << ltot-i << " " << myw.dot(ground) << endl;
            }
            myw.setzero(stream[0]);
            myw.mul(sys.Ham, fn_CPU, ground, siteCPU.Ham, env.Ham, 1.0, "inii", sysbasis, envbasis, stream[0]);
            mout << ltot-i-1 << " " << myw.dot(ground) << endl;
        }
        
        sys.clear();
        wave_store[0].copy(ground, stream[1]);

        start = high_resolution_clock::now();
        
        reducematrix rou;
        rou.set(wavetorou(ground, 'e', stream[0]),stream[0]);
        th1.join();
        cublasSetStream(GlobalHandle, 0);
        envtrun=routotrunc(rou, i+1==lx*ly/2);
        rou.clear();
        cudaDeviceSynchronize();
        ground.clear();

        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << endl << "trun time=" << duration.count() << "ms";
        time_3+=duration.count()/1000.0;

        start = high_resolution_clock::now();
        addonesite_replace(env, envbasis, envtrun, 'e');
        
        wave_store[0].toGPU(ground, 0);

        maptodisk(envbasis, file+"envmap"+to_string(i+1));
        if (i<ltot-end-1) {
            remove((file+"sysbl"+to_string(ltot-i-2)).c_str());
            remove((file+"systrun"+to_string(ltot-i-1)).c_str());
        }
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << ", newH time=" << duration.count() << "ms";
        time_4+=duration.count()/1000.0;

        start = high_resolution_clock::now();
        // th0.join();
        cout << ", sys memCPU=" << sys_pre.mem_size()/1024 << "GB";

        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        cout << ", res time=" << duration.count() << "ms" << endl;
    }
    envtrun.todisk(file+"envtrun"+to_string(ltot-end), 'n');
    env.optodisk(file+"envbl"+to_string(ltot-end));
    if (mout.is_open()) {
        mout << endl;
        mout.close(); 
    }
    cout << "prep time=" << time_1 << "s" << endl;
    cout << "lanc time=" << time_2 << "s" << endl;
    cout << "trun time=" << time_3 << "s" << endl;
    cout << "newH time=" << time_4 << "s" << endl;
    for (size_t i = 0; i < 2; i++) {
        cudaStreamDestroy(stream[i]);
    }
}

void savewave() {
    ofstream out(file+"wave", ios::out | ios::binary | ios::trunc);
    ground.todisk(out, 'n');
    out.close();
}

void readwave() {
    ifstream in(file+"wave", ios::in | ios::binary);
    ground.fromdisk(in, 'n', 0);
    in.close();
}
