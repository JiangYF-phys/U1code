#include "truncation.hpp"

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    return idx;
}

reducematrix wavetorou(const wave &myw, char side, cudaStream_t stream) {    
    // cout << side << endl;
    if (side=='s') {
        return myw.reducematrix::wavemul(myw, 'n', 't', stream);
    } else {
        assert(side=='e');
        return myw.reducematrix::wavemul(myw, 't', 'n', stream);
    }
}

vector<int> keephelp(const vector<vector<double> > &val, int num) {
    int i=0;
    while (num>=0 && i<int(val.size())) {
        num-=val[i].size();
        i++;
    }
    num+=val[i-1].size();
    vector<int> dat(2);
    dat[0]=i-1;dat[1]=num;
    return dat;
}

vector<int> sorteig(const vector<vector<double> > &val) {
    vector<int> keep(val.size());
    vector<double> totv;
    double entropy=0.0;
    for (size_t i=0; i<val.size(); ++i) {
        for (size_t j=0; j< val[i].size(); ++j) {
            totv.push_back( (val[i])[j] );
            if ((val[i])[j]>pow(10.0,-15)) {
                entropy+=-(val[i])[j]*log((val[i])[j]);
            }
        }
    }
    
    int count=0, myk=0;
    double rest=1.0, err=0.0;
    for (auto i: sort_indexes(totv) ) {
        rest-=totv[i];
        if (count<=kept_min || (count<=kept_max && rest>trun_error) ) {
            if (totv[i]>pow(10.0,-16)) {
                vector<int> loc=keephelp(val, int(i));
                if (keep[loc[0]]<=loc[1]+1) {
                    keep[loc[0]] = loc[1]+1;
                }
                err=rest;
                myk=count;
            }
        }
        count++;
    }
    
    cout << "trun_err=" << err << ", entropy=" << entropy << ", kept=" << myk << ", ";
	string name="out/energy.dat";
	ofstream out(name.c_str(), ios::out | ios::app);
	if (out.is_open()) {
        out << scientific;
        out << "entropy=" << entropy << endl;
        out << "trun_err=" << err  << ", kept=" << myk << ", ";
		out.close();
	}
    return keep;
}

reducematrix routotrunc(const reducematrix &rou, bool spectrumflag) {
    vector<vector<double> > val;
    vector<mblock*> vec;
    int lv=0; double tot=0;

    // cusolverDnHandle_t cusolverH;
    
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    int* devInfo;
    cudaMalloc((void**)&devInfo, sizeof(int));
    double *devW, *d_work;
    for (int i=0; i<rou.size(); ++i) {
		double* myval = new double[rou.getsl(i)] ();
		
        mblock* myvec=new mblock(rou.get(i), 0);
		myvec->mul_num(-1,0);

        int lwork=0;
        cudaMalloc((void**)&devW, rou.getsl(i)*sizeof(double));
        cusolverDnDsyevd_bufferSize(GlobalDnHandle, jobz, uplo, rou.getsl(i), myvec->mat, rou.getsl(i), devW, &lwork);
        cudaMalloc((void**)&d_work, sizeof(double)*lwork);
        
        cusolverDnDsyevd(GlobalDnHandle, jobz, uplo, rou.getsl(i), myvec->mat, rou.getsl(i), devW, d_work, lwork, devInfo);
        cudaDeviceSynchronize();

        cudaMemcpy(myval, devW, sizeof(double)*rou.getsl(i), cudaMemcpyDeviceToHost);

        cudaFree(devW);
        cudaFree(d_work);
		vector<double> tmp; tmp.clear();
		for (int j = 0; j < rou.getsl(i); ++j){
			tmp.push_back(-myval[j]);
            tot+=(-myval[j]);
		}
		delete [] myval; myval=NULL;
        val.push_back(tmp);
        lv+=tmp.size();
        vec.push_back(myvec);
    }
    cudaFree(devInfo);
    
    // cout << lv << ", " << tot << endl;

	if (spectrumflag) {
		string name1 = "out/spectrum.dat";
		ofstream out1(name1.c_str(), ios::out | ios::trunc);
		for (int i = 0; i < rou.size(); ++i) {
			out1 << "S&N= " << rou.getjl(i) << " " << rou.getnl(i) << endl;
			for (int j = 0; j < rou.getsl(i); ++j) {
				out1 << val[i][j] << endl;
			}
		}
		out1.close();
	}
    
    vector<int> keep=sorteig(val);
    reducematrix trunc(0,1);
    int kept=0;
    // cout << endl;
    for (int i=0; i<rou.size(); ++i) {
        if (keep[i]>0) {
            // cout << val[i][keep[i]-1] << ", ";
            mblock tmp;
			tmp=vec[i]->block(0,0,vec[i]->sleft,keep[i]);
			tmp.set(rou.get(i));
            trunc.add(tmp);
            kept+=keep[i];
        }
    }
    // cout << endl;
    // cout << "total states=" << kept;
	// string name="out/energy.dat";
	// ofstream out(name.c_str(), ios::out | ios::app);
	// if (out.is_open()) {
    //     out << scientific;
	// 	out << "total states=" << kept;
	// 	out.close();
	// }

    for (size_t i = 0; i < vec.size(); i++) {
        cudaFree(vec[i]->mat);
    }
    
    return trunc;
}
