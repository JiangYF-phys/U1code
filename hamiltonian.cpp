#include "hamiltonian.hpp"
#include "global.hpp"

Hamilton::Hamilton(int l, vector<int> ls, vector<char> type) {
    myl=l;
    opl=ls;
    stype = type;
    op = new reducematrix*[opl.size()] ();
    for (size_t i=0; i<opl.size(); ++i) {
        op[i] = new reducematrix[opl[i]] ();
    }
}

Hamilton::Hamilton(const Hamilton& ham) {
    myl=ham.myl;
    opl=ham.opl;
    stype = ham.stype;
	Ham=ham.Ham;
    op = new reducematrix*[opl.size()] ();
    for (size_t seq=0; seq<opl.size(); ++seq) {
        op[seq] = new reducematrix[opl[seq]] ();
		for (int i = 0; i < opl[seq]; ++i) {
			op[seq][i] = ham.op[seq][i];
		}
    }
}

Hamilton::Hamilton(const string filename){
    ifstream in(filename.c_str(), ios::in | ios::binary);
    in.read((char*) (&myl), sizeof(myl));
    Ham.fromdisk(in, 'u', 0);
    int n_op;
	in.read((char*)(&n_op), sizeof(n_op));
	stype.resize(n_op);
	for (size_t seq = 0; seq < n_op; ++seq) {
		in.read((char*)(&stype[seq]), sizeof(stype[seq]));
	}
    opl.resize(n_op);
    for (int seq=0; seq<n_op; ++seq) {
        in.read((char*) (&opl[seq]), sizeof(opl[seq]));
    }
    op = new reducematrix*[n_op] ();
    for (int seq=0; seq<n_op; ++seq) {
        op[seq] = new reducematrix[opl[seq]] ();
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].fromdisk(in, stype[seq], 0);
        }
    }
    in.close();
}

Hamilton::~Hamilton() {
    this->clear();
}

void Hamilton::clear() {
    Ham.clear();
    for (size_t seq=0; seq<opl.size(); ++seq) {
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].clear();
        }
        delete [] op[seq]; op[seq]=NULL;
    }
    delete [] op ; op = NULL;
    opl.clear();
}


int Hamilton::len() const { return myl; }

double Hamilton::mem_size() const {
    double msize=Ham.mem_size();
    for (size_t seq=0; seq<opl.size(); ++seq) {
        for (int i=0; i<opl[seq]; ++i) {
            msize+=op[seq][i].mem_size();
        }
    }
    return msize;
}

Hamilton& Hamilton::operator =(const Hamilton &rhs) {
    if (this != &rhs) {
        for (size_t seq=0; seq<opl.size(); ++seq) {
            for (int i=0; i<opl[seq]; ++i) {
                op[seq][i].clear();
            }
            delete [] op[seq]; op[seq]=NULL;
        }
        delete [] op;
        
        myl=rhs.myl;
        stype = rhs.stype;
        Ham=rhs.Ham;
        opl=rhs.opl;
        op = new reducematrix*[opl.size()] ();
        for (size_t seq=0; seq<opl.size(); ++seq) {
            op[seq] = new reducematrix[opl[seq]] ();
            for (int i=0; i<opl[seq]; ++i) {
                op[seq][i]=rhs.op[seq][i];
            }
        }
    }
    return *this;
}

void Hamilton::fromC(const HamiltonCPU &rhs) {
    this->clear();
    
    myl=rhs.myl;
    stype = rhs.stype;
    Ham.fromC(rhs.Ham, 0);
    opl=rhs.opl;
    op = new reducematrix*[opl.size()] ();
    for (size_t seq=0; seq<opl.size(); ++seq) {
        op[seq] = new reducematrix[opl[seq]] ();
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].fromC(rhs.op[seq][i], 0);
        }
    }
}

void Hamilton::toCPU(HamiltonCPU &rhs) {
    rhs.clear();
    
    rhs.myl=myl;
    rhs.stype=stype;
    Ham.toCPU(rhs.Ham, 0);
    rhs.opl=opl;
    rhs.op = new reducematrixCPU*[rhs.opl.size()] ();
    for (size_t seq=0; seq<rhs.opl.size(); ++seq) {
        rhs.op[seq] = new reducematrixCPU[rhs.opl[seq]] ();
        for (int i=0; i<rhs.opl[seq]; ++i) {
            op[seq][i].toCPU(rhs.op[seq][i], 0);
        }
    }
}


void Hamilton::truncHam(const reducematrix &trunc) {
	Ham = Ham.applytrunc(trunc, 0);
	for (size_t seq = 0; seq < opl.size(); ++seq) {
		for (int i = 0; i < opl[seq]; ++i) {
			op[seq][i] = op[seq][i].applytrunc(trunc, 0);
		}
	}
}

void Hamilton::fromdisk(const string filename){
    this->clear();
    ifstream in(filename.c_str(), ios::in | ios::binary);
    in.read((char*) (&myl), sizeof(myl));
    Ham.fromdisk(in, 'u', 0);
    int n_op;
	in.read((char*)(&n_op), sizeof(n_op));
	stype.resize(n_op);
	for (size_t seq = 0; seq < stype.size(); ++seq) {
		in.read((char*)(&stype[seq]), sizeof(stype[seq]));
	}
    opl.resize(n_op);
    for (int seq=0; seq<opl.size(); ++seq) {
        in.read((char*) (&opl[seq]), sizeof(opl[seq]));
    }
    op = new reducematrix*[opl.size()] ();
    for (int seq=0; seq<opl.size(); ++seq) {
        op[seq] = new reducematrix[opl[seq]] ();
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].fromdisk(in, stype[seq], 0);
        }
    }
    in.close();
}

void Hamilton::optodisk(const string filename) const{
    ofstream out(filename.c_str(), ios::out | ios::binary | ios::trunc);
	out.write((char*) (&myl), sizeof(myl));
    Ham.todisk(out, 'u');
    int n_op = opl.size();
    out.write((char*)(&n_op), sizeof(n_op));
    for (size_t seq = 0; seq < stype.size(); ++seq) {
		out.write((char*)(&stype[seq]), sizeof(stype[seq]));
	}
    for (size_t seq=0; seq<opl.size(); ++seq) {
        out.write((char*) (&opl[seq]), sizeof(opl[seq]));
    }
    for (size_t seq=0; seq<opl.size(); ++seq) {
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].todisk(out, stype[seq]);
        }
    }
    out.close();
}

reducematrix basistoid(const vector<repmap> basis, char lr) {
    reducematrix Ham(0,1);
    vector<vector<int>> tmp;
	if (lr == 'l') {
		for (auto x : basis) {
			vector<int> arr;
			arr.push_back(x.j1);
			arr.push_back(x.n1);
			arr.push_back(x.len);
			vector<vector<int>>::iterator iter = find(tmp.begin(), tmp.end(), arr);
			if (iter == tmp.end()) {
				tmp.push_back(arr);
			}
		}
	}
	else {
		for (auto x : basis) {
			vector<int> arr;
			arr.push_back(x.j2);
			arr.push_back(x.n2);
			arr.push_back(x.len);
			vector<vector<int>>::iterator iter = find(tmp.begin(), tmp.end(), arr);
			if (iter == tmp.end()) {
				tmp.push_back(arr);
			}
		}
	}

	for (auto x : tmp) {
		mblock blk(x[0], x[0], x[2], x[2], x[1], x[1]);
		Ham.add(blk);
	}
	Ham.toidentity();

	return Ham;
}


HamiltonCPU::HamiltonCPU(int l, vector<int> ls, vector<char> intype) {
    myl=l;
    opl=ls;
    stype=intype;
    op = new reducematrixCPU*[opl.size()] ();
    for (size_t i=0; i<opl.size(); ++i) {
        op[i] = new reducematrixCPU[opl[i]] ();
    }
}


HamiltonCPU::HamiltonCPU(const string filename){
    ifstream in(filename.c_str(), ios::in | ios::binary);
    in.read((char*) (&myl), sizeof(myl));
    Ham.fromdisk(in, 'u');
    int n_op;
	in.read((char*)(&n_op), sizeof(n_op));
	stype.resize(n_op);
	for (size_t seq = 0; seq < n_op; ++seq) {
		in.read((char*)(&stype[seq]), sizeof(stype[seq]));
	}
    opl.resize(n_op);
    for (int seq=0; seq<n_op; ++seq) {
        in.read((char*) (&opl[seq]), sizeof(opl[seq]));
    }
    op = new reducematrixCPU*[n_op] ();
    for (int seq=0; seq<n_op; ++seq) {
        op[seq] = new reducematrixCPU[opl[seq]] ();
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].fromdisk(in, stype[seq]);
        }
    }
    in.close();
}

HamiltonCPU::~HamiltonCPU() {
    this-> clear();
}

void HamiltonCPU::clear() {
    Ham.clear();
    for (size_t seq=0; seq<opl.size(); ++seq) {
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].clear();
        }
        delete [] op[seq]; op[seq]=NULL;
    }
    delete [] op ; op = NULL;
    opl.clear();
}

double HamiltonCPU::mem_size() const {
    double msize=Ham.mem_size();
    for (size_t seq=0; seq<opl.size(); ++seq) {
        for (int i=0; i<opl[seq]; ++i) {
            msize+=op[seq][i].mem_size();
        }
    }
    return msize;
}

void HamiltonCPU::fromdisk(const string filename){
    this->clear();
    ifstream in(filename.c_str(), ios::in | ios::binary);
    in.read((char*) (&myl), sizeof(myl));
    Ham.fromdisk(in, 'u');
    int n_op;
	in.read((char*)(&n_op), sizeof(n_op));
	stype.resize(n_op);
	for (size_t seq = 0; seq < n_op; ++seq) {
		in.read((char*)(&stype[seq]), sizeof(stype[seq]));
	}
    opl.resize(n_op);
    for (int seq=0; seq<n_op; ++seq) {
        in.read((char*) (&opl[seq]), sizeof(opl[seq]));
    }
    op = new reducematrixCPU*[n_op] ();
    for (int seq=0; seq<n_op; ++seq) {
        op[seq] = new reducematrixCPU[opl[seq]] ();
        for (int i=0; i<opl[seq]; ++i) {
            op[seq][i].fromdisk(in, stype[seq]);
        }
    }
    in.close();
}
