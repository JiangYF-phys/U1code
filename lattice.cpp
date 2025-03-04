#include "lattice.hpp"

bond::bond(vector<double> amp, vector<double> coef, int l1, int l2) {
    this->amp = amp;
	for (size_t i = 0; i < amp.size(); ++i) {
		this->amp[i] *= coef[i];
	}
    this->l1  = l1; this->l2 = l2;
}

bond::~bond() {
    amp.clear();
}

void bond::def(vector<double> amp, vector<double> coef, int l1, int l2) {
    this->amp = amp;
	for (size_t i = 0; i < amp.size(); ++i) {
		this->amp[i] *= coef[i];
	}
    this->l1  = l1; this->l2 = l2;
}

ostream& operator <<(ostream& out, const vector<bond>& lattice) {
    out << endl;
	for (size_t i = 0; i < lattice.size(); ++i) {
		cout << lattice[i].l1 << ", " << lattice[i].l2 << endl;
	}
    return out;
}

void makelattice(vector<bond> &latt) {    
    for (size_t i=0; i<latt.size(); ++i) {
        if (latt[i].l1 > latt[i].l2) {
            int ll=latt[i].l1;
            latt[i].l1=latt[i].l2;
            latt[i].l2=ll;
        }
    }
    string name="out/latt.dat";
    ofstream out(name.c_str(), ios::out);
    out << latt.size() << endl;
    for (size_t i = 0; i < latt.size(); ++i) {
        out << latt[i].l1 << " " << latt[i].l2 << endl;
    }
    out.close();
}

fourblock::fourblock() {}

fourblock::~fourblock() {}

vector<int> sys_env_helper(const vector<bond> &sys_env) {
    vector<int> result(sys_env.size(),0), sys_id(sys_env.size()), env_id(sys_env.size());
    for (size_t i=0; i<sys_env.size(); i++) {
        sys_id[i]=sys_env[i].l1;
        env_id[i]=sys_env[i].l2;
    }
    vector<int>::iterator it;
    sort(sys_id.begin(), sys_id.end());
    it=unique_copy(sys_id.begin(), sys_id.end(), sys_id.begin());
    int sys_size=distance(sys_id.begin(), it);
    sys_id.resize(sys_size);
    
    sort(env_id.begin(), env_id.end());
    it=unique_copy(env_id.begin(), env_id.end(), env_id.begin());
    int env_size=distance(env_id.begin(), it);
    env_id.resize(env_size);
    
    if (sys_size<=env_size) {
        for (int i=0; i<sys_size; i++) {
            int count=0;
            for (size_t j=0; j<sys_env.size(); j++) {
                if (sys_id[i]==sys_env[j].l1) count++;
            }
            if (count>1) {
                for (size_t j=0; j<sys_env.size(); j++) {
                    if (sys_id[i]==sys_env[j].l1) result[j]=i+1;
                }
            }
        }
        for (int i=0; i<env_size; i++) {
            int count=0;
            for (size_t j=0; j<sys_env.size(); j++) {
                if (env_id[i]==sys_env[j].l2 && result[j]==0) count++;
            }
            if (count>1) {
                for (size_t j=0; j<sys_env.size(); j++) {
                    if (env_id[i]==sys_env[j].l2 && result[j]==0) result[j]=-(i+1);
                }
            }
        }
    } else {
        for (int i=0; i<env_size; i++) {
            int count=0;
            for (size_t j=0; j<sys_env.size(); j++) {
                if (env_id[i]==sys_env[j].l2) count++;
            }
            if (count>1) {
                for (size_t j=0; j<sys_env.size(); j++) {
                    if (env_id[i]==sys_env[j].l2) result[j]=-(i+1);
                }
            }
        }
        
        for (int i=0; i<sys_size; i++) {
            int count=0;
            for (size_t j=0; j<sys_env.size(); j++) {
                if (sys_id[i]==sys_env[j].l1 && result[j]==0) count++;
            }
            if (count>1) {
                for (size_t j=0; j<sys_env.size(); j++) {
                    if (sys_id[i]==sys_env[j].l1 && result[j]==0) result[j]=i+1;
                }
            }
        }
    }
    
    return result;
}

void fourblock::set(int l1, int ltot, const vector<bond>& lattice, int seq) {
	sys_len=l1;
	env_len=ltot-l1-2;
	int* mylatt=new int[ltot] ();
	if (sys_len>0 && env_len>0) {
		for (size_t i = 0; i < lattice.size(); ++i) {
            if (lattice[i].amp[seq] != 0) {
                if (lattice[i].l1<sys_len) {
                    if (lattice[i].l2==sys_len) {
                        sys_st1.push_back(lattice[i]);
                        mylatt[lattice[i].l1]=1;
                    }
                    if (lattice[i].l2==sys_len+1) {
                        sys_st2.push_back(lattice[i]);
                        mylatt[lattice[i].l1]=1;
                    }
                    if (lattice[i].l2>sys_len+1) {
                        sys_env.push_back(lattice[i]);
                        mylatt[lattice[i].l1]=1;mylatt[lattice[i].l2]=1;
                    }
                }
                if (lattice[i].l1==sys_len) {
                    if (lattice[i].l2==sys_len+1) {
                        st1_st2.push_back(lattice[i]);
                        mylatt[lattice[i].l2]=1;
                    }
                    if (lattice[i].l2>sys_len+1) {
                        st1_env.push_back(lattice[i]);
                        mylatt[lattice[i].l2]=1;
                    }
                }
                if (lattice[i].l1==sys_len+1 && lattice[i].l2>sys_len+1) {
                    st2_env.push_back(lattice[i]);
                    mylatt[lattice[i].l2]=1;
                }
            }
		}
        
		for (int i = 0; i < sys_len; ++i) {
			if (mylatt[i]==1) {
				sys_idx.push_back(i);
			}
		}
		for (int i = sys_len+2; i < ltot; ++i) {
			if (mylatt[i]==1) {
				env_idx.push_back(i);
			}
		}
	}
	delete [] mylatt;
	mylatt=NULL;
}
