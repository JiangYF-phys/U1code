#include "wavefunc.hpp"
#include "global.hpp"

double wave::norm() const {
    double nor=0;
    for (int i=0; i<size(); ++i) {
        nor+=pow(get(i).norm(),2);
    }
    return sqrt(nor);
}

double wave::normalize(cudaStream_t stream) {
    double nor;
    nor=1/norm();
    for (int i=0; i<size(); ++i) {
        num_mul_block(i, nor, stream);
    }
    return 1/nor;
}

double wave::dot(const wave &block1) const {
    double inner=0;
    for (int i=0; i< size(); ++i) {
        // inner+=cblas_ddot(getsl(i)*getsr(i), get(i).mat, 1, block1.get(i).mat, 1);
        double nor=0;
        cublasDdot(GlobalHandle,getsl(i)*getsr(i),get(i).mat,1,block1.get(i).mat,1,&nor);
        inner+=nor;
    }
    return inner;
}

void wave::initial(const vector<repmap> &sys_map, const vector<repmap> &env_map, int snum, int fnum) {
    this->sign(1);
    int sysj, sysn, envj, envn;
    sysj=-77777;sysn=-77777;
    for (size_t i=0; i<sys_map.size(); ++i) {
        if (sys_map[i].j!=sysj || sys_map[i].n!=sysn) {
            sysj=sys_map[i].j;sysn=sys_map[i].n;
            envj=-77777;envn=-77777;
            for (size_t j=0; j<env_map.size(); ++j) {
                if (env_map[j].j!=envj || env_map[j].n!=envn) {
                    envj=env_map[j].j;envn=env_map[j].n;
                    if (sysj + envj == snum && sysn + envn==fnum) {
                        mblock tmp(sysj, envj, sys_map[i].end, env_map[j].end, sysn, envn);
                        cudaMemset(tmp.mat, 0, sizeof(double)*(sys_map[i].end)*(env_map[j].end));
                        this->add( tmp );
                    }
                }
            }
        }
    }
}

void wave::flipsyssite(const vector<repmap> &sys_map, const wave &myw) {
    for (size_t wav_i = 0; wav_i < myw.size(); wav_i++) {
        int old_sys_j=myw.getjl(wav_i);
        int old_sys_n=myw.getnl(wav_i);
        int partA[4], partB[4];
        for (size_t wav_j = 0; wav_j < this->size(); wav_j++) {
            if ( this->getjl(wav_j)==old_sys_j+2 && this->getnl(wav_j)==old_sys_n) {
                for (size_t sys_i = 0; sys_i < sys_map.size(); sys_i++) {
                    if ( sys_map[sys_i].j==old_sys_j && sys_map[sys_i].n==old_sys_n && sys_map[sys_i].j2==-1 ) {
                        int loc=searchmap(sys_map, sys_map[sys_i].j1, 1, sys_map[sys_i].j1 + 1, sys_map[sys_i].n1, sys_map[sys_i].n2, sys_map[sys_i].n);
                        if (loc>-1) {
                            if (sys_map[sys_i].len==sys_map[loc].len && getsr(wav_j)==myw.getsr(wav_i)) {
                                partA[0]=sys_map[loc].bgn; partA[1]=sys_map[loc].len;
                                partA[2]=0; partA[3]=getsr(wav_j);

                                partB[0]=sys_map[sys_i].bgn; partB[1]=sys_map[sys_i].len;
                                partB[2]=0; partB[3]=myw.getsr(wav_i);
                                this->setblockpart(wav_j, partA, myw.get(wav_i), partB, 0);
                            } else {
                                cout << "error: flip spin" << endl;
                            }
                        }
                    }
                }
            }
        }
    }
}


mul_store::mul_store(int mul_i, int mul_j, int mul_nl, int mul_nr, int mul_il, int mul_ir, int mul_jl, int mul_jr, double mul_9j, double mul_mat) {
    this->mul_i=mul_i; this->mul_j=mul_j;
    this->mul_nl=mul_nl; this->mul_nr=mul_nr;
    this->mul_il=mul_il; this->mul_ir=mul_ir;
    this->mul_jl=mul_jl; this->mul_jr=mul_jr;
    this->mul_9j=mul_9j; this->mul_mat=mul_mat;
}

mul_store::~mul_store() {
}

vector<mul_store> sysmulhelper(const reducematrix &block1, const reducematrixCPU &block2, const char &flag1, const char &flag2, const vector<repmap> &mymap) {
    vector<mul_store> store;
    store.clear();
    for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
        int b1_nl_t, b1_nr_t, b1_jl_t, b1_jr_t;
        if (flag1=='t') {
            b1_nl_t = block1.getnr(sys_i); b1_nr_t = block1.getnl(sys_i); b1_jl_t = block1.getjr(sys_i); b1_jr_t = block1.getjl(sys_i);
        } else {
            b1_nl_t = block1.getnl(sys_i); b1_nr_t = block1.getnr(sys_i); b1_jl_t = block1.getjl(sys_i); b1_jr_t = block1.getjr(sys_i);
        }
        for (int sys_j=0; sys_j<block2.size(); ++sys_j) {
            int b2_nl_t, b2_nr_t, b2_jl_t, b2_jr_t;
            if (flag2=='t') {
                b2_nl_t = block2.mat[sys_j]->nright; b2_nr_t = block2.mat[sys_j]->nleft; b2_jl_t = block2.mat[sys_j]->jright; b2_jr_t = block2.mat[sys_j]->jleft;
            } else {
                b2_nl_t = block2.mat[sys_j]->nleft; b2_nr_t = block2.mat[sys_j]->nright; b2_jl_t = block2.mat[sys_j]->jleft; b2_jr_t = block2.mat[sys_j]->jright;
            }
            int sys_nl=b1_nl_t + b2_nl_t;  int sys_nr=b1_nr_t + b2_nr_t;
            int sys_jl = b1_jl_t + b2_jl_t;
            int sys_il=searchmap(mymap, b1_jl_t, b2_jl_t, sys_jl, b1_nl_t, b2_nl_t, sys_nl);
            if (sys_il>=0) {
                int sys_jr = b1_jr_t + b2_jr_t;
                int sys_ir=searchmap(mymap, b1_jr_t, b2_jr_t, sys_jr, b1_nr_t, b2_nr_t, sys_nr);
                if (sys_ir>=0) {
                    double sys_9j=pow(block2.sign(),b1_nr_t % 2);
                    mul_store tmp(sys_i, sys_j, sys_nl, sys_nr, sys_il, sys_ir, sys_jl, sys_jr, sys_9j, block2.mat[sys_j]->mat[0]);
                    store.push_back(tmp);
                }
            }
        }
    }
    return store;
}

vector<mul_store> envmulhelper(const reducematrixCPU &block1, const reducematrix &block2, const char &flag1, const char &flag2, const vector<repmap> &mymap) {
    vector<mul_store> store;
    store.clear();
    for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
        int b1_nl_t, b1_nr_t, b1_jl_t, b1_jr_t;
        if (flag1=='t') {
            b1_nl_t = block1.mat[sys_i]->nright; b1_nr_t = block1.mat[sys_i]->nleft; b1_jl_t = block1.mat[sys_i]->jright; b1_jr_t = block1.mat[sys_i]->jleft;
        } else {
            b1_nl_t = block1.mat[sys_i]->nleft; b1_nr_t = block1.mat[sys_i]->nright; b1_jl_t = block1.mat[sys_i]->jleft; b1_jr_t = block1.mat[sys_i]->jright;
        }
        for (int sys_j=0; sys_j<block2.size(); ++sys_j) {
            int b2_nl_t, b2_nr_t, b2_jl_t, b2_jr_t;
            if (flag2=='t') {
                b2_nl_t = block2.getnr(sys_j); b2_nr_t = block2.getnl(sys_j); b2_jl_t = block2.getjr(sys_j); b2_jr_t = block2.getjl(sys_j);
            } else {
                b2_nl_t = block2.getnl(sys_j); b2_nr_t = block2.getnr(sys_j); b2_jl_t = block2.getjl(sys_j); b2_jr_t = block2.getjr(sys_j);
            }
            int sys_nl=b1_nl_t + b2_nl_t;  int sys_nr=b1_nr_t + b2_nr_t;
            int sys_jl = b1_jl_t + b2_jl_t;
            int sys_il=searchmap(mymap, b1_jl_t, b2_jl_t, sys_jl, b1_nl_t, b2_nl_t, sys_nl);
            if (sys_il>=0) {
                int sys_jr = b1_jr_t + b2_jr_t;
                int sys_ir=searchmap(mymap, b1_jr_t, b2_jr_t, sys_jr, b1_nr_t, b2_nr_t, sys_nr);
                if (sys_ir>=0) {
                    double sys_9j=pow(block2.sign(),b1_nr_t % 2);
                    mul_store tmp(sys_i, sys_j, sys_nl, sys_nr, sys_il, sys_ir, sys_jl, sys_jr, sys_9j, block1.mat[sys_i]->mat[0]);
                    store.push_back(tmp);
                }
            }
        }
    }
    return store;
}


void wave::mul_help(const reducematrix &block1, const reducematrixCPU &block2, const mblock &myw, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], double* tmp_mat, const vector<repmap> &sys_map, const vector<repmap> &env_map, const vector<mul_store> &sys_store, const vector<mul_store> &env_store, cudaStream_t stream) {
    
    for (size_t sys_i=0; sys_i<sys_store.size(); ++sys_i) {
        bool cdt1=sys_store[sys_i].mul_jr==myw.jleft && sys_store[sys_i].mul_nr==myw.nleft;
        if (cdt1) {
            for (size_t env_i=0; env_i<env_store.size(); ++env_i) {
                bool cdt2;
                cdt2=env_store[env_i].mul_jr==myw.jright && env_store[env_i].mul_nr==myw.nright;
                if (cdt2) {
                    double alpha = para * sys_store[sys_i].mul_9j * env_store[env_i].mul_9j;
                    alpha *= pow(block3.sign()*block4.sign(), myw.nleft % 2);
                    int loc=this->search(sys_store[sys_i].mul_jl, env_store[env_i].mul_jl, sys_store[sys_i].mul_nl, env_store[env_i].mul_nl);
                    assert(loc!=-1);
                    int bgn[4]={sys_map[sys_store[sys_i].mul_il].bgn,sys_map[sys_store[sys_i].mul_ir].bgn,env_map[env_store[env_i].mul_il].bgn,env_map[env_store[env_i].mul_ir].bgn};
                    this->mult_subblock_subblock_rank(loc, alpha, block1.get(sys_store[sys_i].mul_i), sys_store[sys_i].mul_mat, myw, env_store[env_i].mul_mat, block4.get(env_store[env_i].mul_j), tmp_mat, bgn, flag, stream);
                    // cout << "loc= " << loc << endl;
                    // cout << this->get(loc) ;
                }
            }
        }
    }
}

void wave::mul(const reducematrix &block1, const reducematrixCPU &block2, const wave &myw, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &sys_map, const vector<repmap> &env_map, cudaStream_t stream) {
    vector<mul_store> sys_store, env_store;
    sys_store=sysmulhelper(block1, block2, flag[0], flag[1], sys_map);
    env_store=envmulhelper(block3, block4, flag[2], flag[3], env_map);

    double* tmp_mat;
    if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
        int s1=0, s3=0;
        for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
            int b1_s;
            if (flag[0]=='t') {
                b1_s = block1.getsr(sys_i);
            } else {
                b1_s = block1.getsl(sys_i);
            }
            if (b1_s>s1) { s1=b1_s; }
        }
        for (int env_i=0; env_i<block4.size(); ++env_i) {
            int b4_s;
            if (flag[3]=='t') {
                b4_s = block4.getsl(env_i);
            } else {
                b4_s = block4.getsr(env_i);
            }
            if (b4_s>s3) { s3=b4_s; }
        }
        cudaMalloc((void**)&tmp_mat, s1*s3*sizeof(double));
    }

    for (int wav_i=0; wav_i<myw.size(); ++wav_i) {
        this->mul_help(block1, block2, myw.get(wav_i), block3, block4, para, flag, tmp_mat, sys_map, env_map, sys_store, env_store, stream);
        // cout << this->get(3) << endl;
    }

    if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
        cudaFree(tmp_mat);
    }
}

void mul_CtoG(const reducematrix &block1, const reducematrixCPU &block2, const wave_CPU &myw, wave &waveG, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &sys_map, const vector<repmap> &env_map, cudaStream_t stream) {
    vector<mul_store> sys_store, env_store;
    sys_store=sysmulhelper(block1, block2, flag[0], flag[1], sys_map);
    env_store=envmulhelper(block3, block4, flag[2], flag[3], env_map);

    double* tmp_mat;
    if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
        int s1=0, s3=0;
        for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
            int b1_s;
            if (flag[0]=='t') {
                b1_s = block1.getsr(sys_i);
            } else {
                b1_s = block1.getsl(sys_i);
            }
            if (b1_s>s1) { s1=b1_s; }
        }
        for (int env_i=0; env_i<block4.size(); ++env_i) {
            int b4_s;
            if (flag[3]=='t') {
                b4_s = block4.getsl(env_i);
            } else {
                b4_s = block4.getsr(env_i);
            }
            if (b4_s>s3) { s3=b4_s; }
        }
        cudaMalloc((void**)&tmp_mat, s1*s3*sizeof(double));
    }

    for (int wav_i=0; wav_i<myw.mat.size(); ++wav_i) {
        mblock tmp;
        tmp=myw.toGPU(wav_i, stream);
        waveG.mul_help(block1, block2, tmp, block3, block4, para, flag, tmp_mat, sys_map, env_map, sys_store, env_store, stream);
    }

    if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
        cudaFree(tmp_mat);
    }
}



// vector<mul_store> mulhelper(const reducematrix &block1, const reducematrix &block2, const char &flag1, const char &flag2, const vector<repmap> &mymap) {
//     vector<mul_store> store;
//     store.clear();
//     for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
//         int b1_nl_t, b1_nr_t, b1_jl_t, b1_jr_t;
//         if (flag1=='t') {
//             b1_nl_t = block1.getnr(sys_i); b1_nr_t = block1.getnl(sys_i); b1_jl_t = block1.getjr(sys_i); b1_jr_t = block1.getjl(sys_i);
//         } else {
//             b1_nl_t = block1.getnl(sys_i); b1_nr_t = block1.getnr(sys_i); b1_jl_t = block1.getjl(sys_i); b1_jr_t = block1.getjr(sys_i);
//         }
//         for (int sys_j=0; sys_j<block2.size(); ++sys_j) {
//             int b2_nl_t, b2_nr_t, b2_jl_t, b2_jr_t;
//             if (flag2=='t') {
//                 b2_nl_t = block2.getnr(sys_j); b2_nr_t = block2.getnl(sys_j); b2_jl_t = block2.getjr(sys_j); b2_jr_t = block2.getjl(sys_j);
//             } else {
//                 b2_nl_t = block2.getnl(sys_j); b2_nr_t = block2.getnr(sys_j); b2_jl_t = block2.getjl(sys_j); b2_jr_t = block2.getjr(sys_j);
//             }
//             int sys_nl=b1_nl_t + b2_nl_t;  int sys_nr=b1_nr_t + b2_nr_t;
//             int sys_jl = b1_jl_t + b2_jl_t;
//             int sys_il=searchmap(mymap, b1_jl_t, b2_jl_t, sys_jl, b1_nl_t, b2_nl_t, sys_nl);
//             if (sys_il>=0) {
//                 int sys_jr = b1_jr_t + b2_jr_t;
//                 int sys_ir=searchmap(mymap, b1_jr_t, b2_jr_t, sys_jr, b1_nr_t, b2_nr_t, sys_nr);
//                 if (sys_ir>=0) {
//                     double sys_9j=pow(block2.sign(),b1_nr_t % 2);
//                     mul_store tmp(sys_i, sys_j, sys_nl, sys_nr, sys_il, sys_ir, sys_jl, sys_jr, sys_9j);
//                     store.push_back(tmp);
//                 }
//             }
//         }
//     }
//     return store;
// }

// void wave::mul(const reducematrix &block1, const reducematrix &block2, const wave &myw, const reducematrix &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &sys_map, const vector<repmap> &env_map, cudaStream_t stream) {
//     vector<mul_store> sys_store, env_store;
//     sys_store=mulhelper(block1, block2, flag[0], flag[1], sys_map);
//     env_store=mulhelper(block3, block4, flag[2], flag[3], env_map);
    
//     if (flag[1] != 'i') {
//         for (size_t sys_i=0; sys_i<sys_store.size(); ++sys_i) {
//             cublasGetMatrixAsync(1, 1, sizeof(double), block2.get(sys_store[sys_i].mul_j).mat, 1, &b2vec[0]+sys_i, 1, stream);
//         }
//     }
//     if (flag[2] != 'i') {
//         for (size_t env_i=0; env_i<env_store.size(); ++env_i) {
//             cublasGetMatrixAsync(1, 1, sizeof(double), block3.get(env_store[env_i].mul_i).mat, 1, &b3vec[0]+env_i, 1, stream);
//         }
//     }

//     double* tmp_mat;
//     if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
//         int s1=0, s3=0;
//         for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
//             int b1_s;
//             if (flag[0]=='t') {
//                 b1_s = block1.getsr(sys_i);
//             } else {
//                 b1_s = block1.getsl(sys_i);
//             }
//             if (b1_s>s1) { s1=b1_s; }
//         }
//         for (int env_i=0; env_i<block4.size(); ++env_i) {
//             int b4_s;
//             if (flag[3]=='t') {
//                 b4_s = block4.getsl(env_i);
//             } else {
//                 b4_s = block4.getsr(env_i);
//             }
//             if (b4_s>s3) { s3=b4_s; }
//         }
//         cudaMalloc((void**)&tmp_mat, s1*s3*sizeof(double));
//     }
//     cudaStreamSynchronize(stream);

//     for (size_t sys_i=0; sys_i<sys_store.size(); ++sys_i) {
//         for (int wav_i=0; wav_i<myw.size(); ++wav_i) {
//             bool cdt1=sys_store[sys_i].mul_jr==myw.getjl(wav_i) && sys_store[sys_i].mul_nr==myw.getnl(wav_i);
//             if (cdt1) {
//                 for (size_t env_i=0; env_i<env_store.size(); ++env_i) {
//                     bool cdt2;
//                     cdt2=env_store[env_i].mul_jr==myw.getjr(wav_i) && env_store[env_i].mul_nr==myw.getnr(wav_i);
//                     if (cdt2) {
//                         double alpha = para * sys_store[sys_i].mul_9j * env_store[env_i].mul_9j;
//                         alpha *= pow(block3.sign()*block4.sign(), myw.getnl(wav_i) % 2);
//                         int loc=this->search(sys_store[sys_i].mul_jl, env_store[env_i].mul_jl, sys_store[sys_i].mul_nl, env_store[env_i].mul_nl);
//                         assert(loc!=-1);
//                         int bgn[4]={sys_map[sys_store[sys_i].mul_il].bgn,sys_map[sys_store[sys_i].mul_ir].bgn,env_map[env_store[env_i].mul_il].bgn,env_map[env_store[env_i].mul_ir].bgn};
//                         this->mult_subblock_subblock_rank(loc, alpha, block1.get(sys_store[sys_i].mul_i), b2vec[sys_i], myw.get(wav_i), b3vec[env_i], block4.get(env_store[env_i].mul_j), tmp_mat, bgn, flag, stream);
//                     }
//                 }
//             }
//         }
//     }
//     if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
//         cudaFree(tmp_mat);
//     }
// }

bool transhelp(const wave &myw, const int &j1, const int &j2, const int &n1, const int &n2, int &loc) {
    for (int i=0; i<myw.size(); ++i) {
        if (myw.getjl(i)==j1 && myw.getjr(i)==j2 && myw.getnl(i)==n1 && myw.getnr(i)==n2) {
            loc=i;
            return false;
        }
    }
    return true;
}

void wave::transLtoR(const wave &myw, const reducematrix &systrun, const reducematrix &envtrun, const vector<repmap> sysmap, const vector<repmap> envmap){
    for (int i=0; i<myw.size(); ++i) {
        for (int j=0; j<systrun.size(); ++j) {
            if ( checktime(systrun.get(j), myw.get(i),'t','n') ) {
                mblock tmp;
                tmp=mconj(systrun.get(j),1.0)* myw.get(i);
                for (size_t l=0; l<envmap.size(); ++l) {
                    if (envmap[l].n==tmp.nright && envmap[l].j==tmp.jright) {
                        int j12 = tmp.jleft + envmap[l].j1;
                        mblock newtmp(tmp.block(0, envmap[l].bgn, tmp.sleft, envmap[l].len));
                        newtmp.set(j12, envmap[l].j2, tmp.nleft+envmap[l].n1, envmap[l].n2);
                        int i1=searchmap(sysmap, tmp.jleft, envmap[l].j1, j12, tmp.nleft, envmap[l].n1, tmp.nleft+envmap[l].n1);
                        for (int k=0; k<envtrun.size(); ++k) {
                            if ( checktime(newtmp, envtrun.get(k),'n','t') && i1>=0) {
                                int loc=this->search(newtmp.jleft, newtmp.jright, newtmp.nleft, newtmp.nright);
                                if (loc>=0) {
                                    this->addsubblock(loc, sysmap[i1].bgn, 0, sysmap[i1].len, envtrun.getsl(k), newtmp*mconj(envtrun.get(k),1.0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void wave::transRtoL(const wave &myw, const reducematrix &systrun, const reducematrix &envtrun, const vector<repmap> sysmap, const vector<repmap> envmap){
    for (int i=0; i<myw.size(); ++i) {
        for (int j=0; j<envtrun.size(); ++j) {
            if ( checktime(myw.get(i), envtrun.get(j),'n','n') ) {
                mblock tmp;
                tmp=myw.get(i)*envtrun.get(j);//??
                for (size_t l=0; l<sysmap.size(); ++l) {
                    if (sysmap[l].n==tmp.nleft && sysmap[l].j==tmp.jleft) {
                        int j23 = sysmap[l].j2 + tmp.jright;
                        mblock newtmp(tmp.block(sysmap[l].bgn, 0, sysmap[l].len, tmp.sright));
                        newtmp.set(sysmap[l].j1, j23, sysmap[l].n1, tmp.nright+sysmap[l].n2);
                        int i1=searchmap(envmap, sysmap[l].j2, tmp.jright, j23, sysmap[l].n2, tmp.nright, tmp.nright+sysmap[l].n2);
                        for (int k=0; k<systrun.size(); ++k) {
                            if ( checktime(systrun.get(k), newtmp,'n','n') && i1>=0) {
                                int loc=this->search(newtmp.jleft, newtmp.jright, newtmp.nleft, newtmp.nright);
                                if (loc>=0) {
                                    this->addsubblock(loc, 0, envmap[i1].bgn, systrun.getsl(k), envmap[i1].len,  systrun.get(k)*newtmp );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


wave_CPU::wave_CPU() {
    tot_size=0;
    // val = new double[tot_size] ();
    cudaMallocHost(&val, tot_size*sizeof(double));
}

wave_CPU::~wave_CPU() {
    this->clear();
}

void wave_CPU::clear() {
    tot_size=0;
    // delete [] val; val=NULL;
    cudaFreeHost(val);
    
    for (size_t i=0; i<mat.size(); ++i) {
        mat[i]->clear();
        // cudaFree(mat[i]->mat);
    }
    mat.clear();
}

void wave_CPU::setzero() {
    memset(val, 0, tot_size*sizeof(double));
}

void wave_CPU::construct(const wave &myw) {
    // this->clear();
    int totsize=0;
    for (size_t i=0; i<myw.size(); ++i) {
        mblockCPU* tmp = new mblockCPU();
        tmp->jleft  = myw.getjl(i); tmp->jright = myw.getjr(i);
        tmp->sleft  = myw.getsl(i); tmp->sright = myw.getsr(i);
        tmp->nleft  = myw.getnl(i); tmp->nright = myw.getnr(i);
        
        // mblock.mat is empty
        mat.push_back(tmp);
        totsize+=tmp->sleft*tmp->sright;
    }
    // val = new double[totsize] ();
    cudaMallocHost(&val, totsize*sizeof(double));
    tot_size=totsize;
}

void wave_CPU::copy(const wave &myw, cudaStream_t stream) {
    int anchor=0;
    // cout << "totsize= " << tot_size << endl;
    for (size_t i=0; i<myw.size(); ++i) {
        int mysize=mat[i]->sleft * mat[i]->sright;
        // cout << mysize << endl;
        cudaMemcpyAsync(&val[0]+anchor,myw.get(i).mat,mysize*sizeof(double),cudaMemcpyDeviceToHost,stream);
        anchor+=mysize;
        // cout << i << ", " << anchor << endl;
    }
}

void wave_CPU::copyval(int loc, const mblock &myw, cudaStream_t stream) {
    int anchor=0;
    for (size_t i=0; i<loc; ++i) {
        anchor+=mat[i]->sleft * mat[i]->sright;
    }
    int mysize=mat[loc]->sleft * mat[loc]->sright;
    cudaMemcpyAsync(&val[0]+anchor,myw.mat,mysize*sizeof(double),cudaMemcpyDeviceToHost,stream);

    cudaDeviceSynchronize();
    // for (size_t i=0; i<mat[loc]->sleft; ++i) {
    //     cout << val[anchor+i] << " ";
    // }
    // cout << endl;
}

void wave_CPU::toGPU(wave &myW, cudaStream_t stream) const {
    myW.clear();
    myW.sign(1);
    int anchor=0;
    for (size_t i=0; i < this->mat.size(); ++i) {
        mblock* tmp=new mblock();
        tmp->jleft=this->mat[i]->jleft;tmp->jright=this->mat[i]->jright;
        tmp->sleft=this->mat[i]->sleft;tmp->sright=this->mat[i]->sright;
        tmp->nleft=this->mat[i]->nleft;tmp->nright=this->mat[i]->nright;

        int mysize=tmp->sleft * tmp->sright;
        cudaMalloc((void**)&(tmp->mat), mysize*sizeof(double));
        cudaMemcpyAsync(tmp->mat, &val[0]+anchor, mysize*sizeof(double), cudaMemcpyHostToDevice, stream);
        // cudaDeviceSynchronize();
        myW.addC(tmp);
        anchor+=mysize;
        // cout << anchor << endl;
    }
}

mblock wave_CPU::toGPU(int loc, cudaStream_t stream) const {
    mblock tmp;
    int anchor=0;
    for (size_t i=0; i < loc; ++i) {
        anchor+=this->mat[i]->sleft * this->mat[i]->sright;
    }
    tmp.jleft=this->mat[loc]->jleft;tmp.jright=this->mat[loc]->jright;
    tmp.sleft=this->mat[loc]->sleft;tmp.sright=this->mat[loc]->sright;
    tmp.nleft=this->mat[loc]->nleft;tmp.nright=this->mat[loc]->nright;
    int mysize=this->mat[loc]->sleft * this->mat[loc]->sright;
    cudaMalloc((void**)&(tmp.mat), mysize*sizeof(double));
    cudaMemcpyAsync(tmp.mat, &val[0]+anchor, mysize*sizeof(double), cudaMemcpyHostToDevice, stream);
    return tmp;
}

void wave_CPU::mul_num(const double& alpha) {
    cblas_dscal(tot_size, alpha, val, 1);
}

void wave_CPU::mul_add(const double& alpha, const wave_CPU& wav) {
    cblas_daxpy(tot_size, alpha, wav.val, 1, val, 1);
}

double wave_CPU::mem_size() {
    return tot_size*sizeof(double)/1024/1024.0;
}

double wave_CPU::dot(const wave &wav, cudaStream_t stream) const {
    int sizemax=0;
    for (size_t i=0; i < mat.size(); ++i) { 
        sizemax=max(sizemax, this->mat[i]->sleft * this->mat[i]->sright); 
    }
    double *tmpmat;
    cudaMalloc((void**)&(tmpmat), sizemax*sizeof(double));

    int anchor=0;
    double inner=0;
    for (size_t i=0; i<mat.size(); ++i) {
        int mysize=mat[i]->sleft * mat[i]->sright;
        cudaMemcpyAsync(tmpmat, &val[0]+anchor, mysize*sizeof(double), cudaMemcpyHostToDevice, stream);
        double nor=0;
        cublasDdot(GlobalHandle,mysize,&tmpmat[0],1,wav.get(i).mat,1,&nor);
        inner+=nor;
        anchor+=mysize;
    }
    cudaFree(tmpmat);
    return inner;
}

double wave_CPU::normalize() {
    double nor;
    nor=cblas_ddot(tot_size, val, 1, val, 1);
    nor=1/sqrt(nor);
    cublasDscal(GlobalHandle,tot_size,&nor,val,1);
    return 1/nor;
}

int wave_CPU::search(const int &jleft, const int &jright, const int &nleft, const int &nright) const {
	vector<mblockCPU*>::const_iterator it = find_if(mat.begin(), mat.end(),[&jleft, &jright, &nleft, &nright] (mblockCPU* const &m) {
		return m->jleft==jleft && m->jright==jright && m->nleft==nleft && m->nright==nright;});
	if (it==mat.end()) {
		return -1;
	} else {
		return distance(mat.begin(), it);
	}
}

void wave_CPU::mul_help(const reducematrix &block1, const reducematrixCPU &block2, const mblock &myw, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], double* tmp_mat, const vector<repmap> &sys_map, const vector<repmap> &env_map, const vector<mul_store> &sys_store, const vector<mul_store> &env_store, cudaStream_t stream) {
    
    mblock tmp;
    // cudaFree(tmp.mat);
    int sizemax=0;
    for (size_t i=0; i < this->mat.size(); ++i) { 
        sizemax=max(sizemax, this->mat[i]->sleft * this->mat[i]->sright); 
    }
    cudaMalloc((void**)&(tmp.mat), sizemax*sizeof(double));
    
    for (size_t sys_i=0; sys_i<sys_store.size(); ++sys_i) {
        bool cdt1=sys_store[sys_i].mul_jr==myw.jleft && sys_store[sys_i].mul_nr==myw.nleft;
        if (cdt1) {
            for (size_t env_i=0; env_i<env_store.size(); ++env_i) {
                bool cdt2;
                cdt2=env_store[env_i].mul_jr==myw.jright && env_store[env_i].mul_nr==myw.nright;
                if (cdt2) {
                    double alpha = para * sys_store[sys_i].mul_9j * env_store[env_i].mul_9j;
                    alpha *= pow(block3.sign()*block4.sign(), myw.nleft % 2);
                    int loc=this->search(sys_store[sys_i].mul_jl, env_store[env_i].mul_jl, sys_store[sys_i].mul_nl, env_store[env_i].mul_nl);
                    assert(loc!=-1);
                    int bgn[4]={sys_map[sys_store[sys_i].mul_il].bgn,sys_map[sys_store[sys_i].mul_ir].bgn,env_map[env_store[env_i].mul_il].bgn,env_map[env_store[env_i].mul_ir].bgn};
                    
                    int anchor=0;
                    for (size_t i=0; i < loc; ++i) { anchor+=this->mat[i]->sleft * this->mat[i]->sright; }

                    tmp.sleft=this->mat[loc]->sleft;
                    tmp.sright=this->mat[loc]->sright;
                    int mysize=this->mat[loc]->sleft * this->mat[loc]->sright;
                    cudaMemcpyAsync(&tmp.mat[0], &val[0]+anchor, mysize*sizeof(double), cudaMemcpyHostToDevice, stream);
                    tmp.mult_subblock_subblock_rank(alpha,block1.get(sys_store[sys_i].mul_i), sys_store[sys_i].mul_mat, myw, env_store[env_i].mul_mat, block4.get(env_store[env_i].mul_j), tmp_mat, bgn, flag, stream);
                    // cudaDeviceSynchronize();
                    this->copyval(loc, tmp, 0);
                    
                }
            }
        }
    }
    // cudaFree(tmp.mat);
}


void mul_GtoC(const reducematrix &block1, const reducematrixCPU &block2, const wave &myw, wave_CPU &waveC, const reducematrixCPU &block3, const reducematrix &block4, const double &para, const char flag[4], const vector<repmap> &sys_map, const vector<repmap> &env_map, cudaStream_t stream) {
    vector<mul_store> sys_store, env_store;
    sys_store=sysmulhelper(block1, block2, flag[0], flag[1], sys_map);
    env_store=envmulhelper(block3, block4, flag[2], flag[3], env_map);

    double* tmp_mat;
    if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
        int s1=0, s3=0;
        for (int sys_i=0; sys_i<block1.size(); ++sys_i) {
            int b1_s;
            if (flag[0]=='t') {
                b1_s = block1.getsr(sys_i);
            } else {
                b1_s = block1.getsl(sys_i);
            }
            if (b1_s>s1) { s1=b1_s; }
        }
        for (int env_i=0; env_i<block4.size(); ++env_i) {
            int b4_s;
            if (flag[3]=='t') {
                b4_s = block4.getsl(env_i);
            } else {
                b4_s = block4.getsr(env_i);
            }
            if (b4_s>s3) { s3=b4_s; }
        }
        cudaMalloc((void**)&tmp_mat, s1*s3*sizeof(double));
    }

    for (int wav_i=0; wav_i<myw.size(); ++wav_i) {
        waveC.mul_help(block1, block2, myw.get(wav_i), block3, block4, para, flag, tmp_mat, sys_map, env_map, sys_store, env_store, stream);
        // cudaDeviceSynchronize();
    }

    if ( (flag[0] != 'i' || flag[1] != 'i') && (flag[2] != 'i' || flag[3] != 'i') ) {
        cudaFree(tmp_mat);
    }
}

// void wave_CPU::constructGPU(wave &myW) {
//     myW.clear();
//     myW.sign(1);
//     int anchor=0;
//     for (size_t i=0; i < this->mat.size(); ++i) {
//         mblock* tmp=new mblock();
//         tmp->jleft=this->mat[i]->jleft;tmp->jright=this->mat[i]->jright;
//         tmp->sleft=this->mat[i]->sleft;tmp->sright=this->mat[i]->sright;
//         tmp->nleft=this->mat[i]->nleft;tmp->nright=this->mat[i]->nright;

//         int mysize=tmp->sleft * tmp->sright;
//         cudaMalloc((void**)&(tmp->mat), mysize*sizeof(double));
//         myW.addC(tmp);
//         anchor+=mysize;
//     }
// }

// void wave_CPU::copyGPU(wave &myw, cudaStream_t stream) {
//     myw.changemat(this->val, stream);
// }

//========================================================================================================================
//                                                      lagacy
//========================================================================================================================
// only appear in measure

void wave::mul(const reducematrix &block1,const wave &myw) {
    for (int i=0; i<myw.size(); ++i) {
        for (int l=0; l<block1.size(); l++) {
            if (block1.getjr(l)==myw.getjl(i) && block1.getnr(l)==myw.getnl(i)) {
                double j9=1.0;
                int loc=this->search(block1.getjl(l), myw.getjr(i), block1.getnl(l), myw.getnr(i));
                assert(loc!=-1);
                this->mult_block_rank(loc,j9,block1.get(l),myw.get(i),'n','n');
            }
        }
    }
}

void wave::mul(const wave &myw, const reducematrix &block2) {
    for (int i=0; i<myw.size(); ++i) {
        for (int r=0; r<block2.size(); ++r) {
            if (block2.getjr(r)==myw.getjr(i) && block2.getnr(r)==myw.getnr(i)) {
                double j9=1.0;
                int loc=this->search(myw.getjl(i), block2.getjl(r), myw.getnl(i), block2.getnl(r));
                assert(loc!=-1);
                this->mult_block_rank(loc,j9,myw.get(i), block2.get(r),'n','t');
            }
        }
    }
}

void wave::mul(const reducematrix &block1,const wave &myw, const reducematrix &block2, const double &para, const char &flag1, const char &flag2) {
    for (int i=0; i < myw.size(); ++i) {
        for (int l=0; l < block1.size(); ++l) {
            int jr1,jl1,nr1,nl1;
            if (flag1=='t') {
                jr1=block1.getjl(l); jl1=block1.getjr(l); nr1=block1.getnl(l); nl1=block1.getnr(l);
            } else {
                jr1=block1.getjr(l); jl1=block1.getjl(l); nr1=block1.getnr(l); nl1=block1.getnl(l);
            }
            if (jr1==myw.getjl(i) && nr1==myw.getnl(i)) {
                for (int r=0; r < block2.size(); ++r) {
                    int jr2,jl2,nr2,nl2;
                    if (flag2=='t') {
                        jr2=block2.getjl(r); jl2=block2.getjr(r); nr2=block2.getnl(r); nl2=block2.getnr(r);
                    } else {
                        jr2=block2.getjr(r); jl2=block2.getjl(r); nr2=block2.getnr(r); nl2=block2.getnl(r);
                    }
                    if (jr2==myw.getjr(i)  && nr2==myw.getnr(i)) {
                        double j9=1.0;
                        j9 *= para * pow(block2.sign(),myw.getnl(i) % 2);
                        int loc=this->search(jl1, jl2, nl1, nl2);
                        assert(loc!=-1);
                        char f2='t';
                        if (flag2=='t') f2='n';
                        this->mult_block_block_rank(loc, j9, block1.get(l),myw.get(i), block2.get(r), flag1, f2);
                    }
                }
            }
        }
    }
}
