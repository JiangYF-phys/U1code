#include "extend.hpp"

Hamilton addonesite(const Hamilton &block, vector<repmap> &basis, char sore) {
	vector<int> ls;
	if (sore=='s') {
        for (int seq=0; seq<num_op; ++seq) {
            ls.push_back(mybase[seq][block.len()+1].sys_idx.size());
        }
	} else {
        for (int seq=0; seq<num_op; ++seq) {
            ls.push_back(mybase[seq][ltot-2-(block.len()+1)].env_idx.size());
        }
	}
    
	Hamilton newblock(block.len()+1,ls, block.stype);
    
    if (sore=='s') {
        newblock.Ham.prod_id(block.Ham, train_site[block.len()+1].Ham, basis, 1.0, 'r', 0);
        newblock.Ham.prod_id(block.Ham, train_site[block.len()+1].Ham, basis, 1.0, 'l', 0);
    }   else {
        newblock.Ham.prod_id(train_site[ltot-block.len()].Ham, block.Ham, basis, 1.0, 'r', 0);
        newblock.Ham.prod_id(train_site[ltot-block.len()].Ham, block.Ham, basis, 1.0, 'l', 0);
    }
    
    // update operators
    if (sore=='s'){
        for (int seq=0; seq<num_op; ++seq) {
            for (size_t i = 0; i < mybase[seq][block.len()+1].sys_idx.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][block.len()].sys_idx.begin(), mybase[seq][block.len()].sys_idx.end(), mybase[seq][block.len()+1].sys_idx[i]);
                if (iter==mybase[seq][block.len()].sys_idx.end()) {
                    newblock.op[seq][i].prod_id(block.Ham, site.op[seq][0], basis, 1.0, 'l', 0);
                } else {
                    int loc=distance(mybase[seq][block.len()].sys_idx.begin(), iter);
                    newblock.op[seq][i].prod_id(block.op[seq][loc], site.Ham, basis, 1.0, 'r', 0);
                }
            }
        }
    } else {
        for (int seq=0; seq<num_op; ++seq) {
            for (size_t i = 0; i < mybase[seq][ltot-2-(block.len()+1)].env_idx.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][ltot-2-block.len()].env_idx.begin(), mybase[seq][ltot-2-block.len()].env_idx.end(), mybase[seq][ltot-2-(block.len()+1)].env_idx[i]);
                if (iter==mybase[seq][ltot-2-block.len()].env_idx.end()) {
                    newblock.op[seq][i].prod_id(site.op[seq][0], block.Ham, basis, 1.0, 'r', 0);
                } else {
                    int loc=distance(mybase[seq][ltot-2-block.len()].env_idx.begin(), iter);
                    newblock.op[seq][i].prod_id(site.Ham, block.op[seq][loc], basis, 1.0, 'l', 0);
                }
            }
        }
    }

    // update Hamiltonian
    if (sore == 's') {
        for (int seq=0; seq<num_op; ++seq) {
            reducematrix tmp(0, site.op[seq][0].sign());
            for (size_t i = 0; i < mybase[seq][block.len()].sys_st1.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][block.len()].sys_idx.begin(), mybase[seq][block.len()].sys_idx.end(), mybase[seq][block.len()].sys_st1[i].l1);
                int loc=distance(mybase[seq][block.len()].sys_idx.begin(), iter);
                tmp.mul_add( mybase[seq][block.len()].sys_st1[i].amp[seq], block.op[seq][loc], 0);
            }
            if (optype[seq]=='c') {
                reducematrix tmp0(0,1);
                tmp0.prod(tmp.conj(0), site.op[seq][0], basis, 1.0, 'n', 'n');
                newblock.Ham+=tmp0;
                newblock.Ham+=tmp0.conj(0);
            } else if (optype[seq]=='n') {
                newblock.Ham.prod(tmp, site.op[seq][0], basis, 1.0, 'n', 'n');
            }
        }
    } else {
        for (int seq=0; seq<num_op; ++seq) {
            reducematrix tmp(0, site.op[seq][0].sign());
            for (size_t i = 0; i < mybase[seq][ltot-2-block.len()].st2_env.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][ltot-2-block.len()].env_idx.begin(), mybase[seq][ltot-2-block.len()].env_idx.end(), mybase[seq][ltot-2-block.len()].st2_env[i].l2);
                int loc=distance(mybase[seq][ltot-2-block.len()].env_idx.begin(), iter);
                tmp.mul_add( mybase[seq][ltot-2-block.len()].st2_env[i].amp[seq], block.op[seq][loc], 0);
            }
            if (optype[seq]=='c') {
                reducematrix tmp0(0,1);
                tmp0.prod(site.op[seq][0].conj(0), tmp, basis, 1.0, 'n', 'n');
                newblock.Ham+=tmp0;
                newblock.Ham+=tmp0.conj(0);
            } else if (optype[seq]=='n') {
                newblock.Ham.prod(site.op[seq][0], tmp, basis, 1.0, 'n', 'n');
            }
        }
    }
    
    return newblock;
}

void addonesite_replace(Hamilton &block, vector<repmap> &basis, const reducematrix &trunc, char sore) {
	vector<int> ls;
	if (sore=='s') {
        for (int seq=0; seq<num_op; ++seq) {
            ls.push_back(mybase[seq][block.len()+1].sys_idx.size());
        }
	} else {
        for (int seq=0; seq<num_op; ++seq) {
            ls.push_back(mybase[seq][ltot-2-(block.len()+1)].env_idx.size());
        }
	}
    
	Hamilton newblock(block.len()+1,ls,block.stype);
    
    if (sore=='s') {
        newblock.Ham.prod_id(block.Ham, train_site[block.len()+1].Ham, basis, 1.0, 'r', 0);
        newblock.Ham.prod_id(block.Ham, train_site[block.len()+1].Ham, basis, 1.0, 'l', 0);
    }   else {
        newblock.Ham.prod_id(train_site[ltot-block.len()].Ham, block.Ham, basis, 1.0, 'r', 0);
        newblock.Ham.prod_id(train_site[ltot-block.len()].Ham, block.Ham, basis, 1.0, 'l', 0);
    }
    
    // update Hamiltonian
    if (sore == 's') {
        for (int seq=0; seq<num_op; ++seq) {
            reducematrix tmp(0, site.op[seq][0].sign());
            for (size_t i = 0; i < mybase[seq][block.len()].sys_st1.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][block.len()].sys_idx.begin(), mybase[seq][block.len()].sys_idx.end(), mybase[seq][block.len()].sys_st1[i].l1);
                int loc=distance(mybase[seq][block.len()].sys_idx.begin(), iter);
                tmp.mul_add( mybase[seq][block.len()].sys_st1[i].amp[seq], block.op[seq][loc], 0);
            }
            if (optype[seq]=='c') {
                reducematrix tmp0(0,1);
                tmp0.prod(tmp.conj(0), site.op[seq][0], basis, 1.0, 'n', 'n');
                newblock.Ham+=tmp0;
                newblock.Ham+=tmp0.conj(0);
            } else if (optype[seq]=='n') {
                newblock.Ham.prod(tmp, site.op[seq][0], basis, 1.0, 'n', 'n');
            }
        }
    } else {
        for (int seq=0; seq<num_op; ++seq) {
            reducematrix tmp(0, site.op[seq][0].sign());
            for (size_t i = 0; i < mybase[seq][ltot-2-block.len()].st2_env.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][ltot-2-block.len()].env_idx.begin(), mybase[seq][ltot-2-block.len()].env_idx.end(), mybase[seq][ltot-2-block.len()].st2_env[i].l2);
                int loc=distance(mybase[seq][ltot-2-block.len()].env_idx.begin(), iter);
                tmp.mul_add( mybase[seq][ltot-2-block.len()].st2_env[i].amp[seq], block.op[seq][loc], 0);
            }
            if (optype[seq]=='c') {
                reducematrix tmp0(0,1);
                tmp0.prod(site.op[seq][0].conj(0), tmp, basis, 1.0, 'n', 'n');
                newblock.Ham+=tmp0;
                newblock.Ham+=tmp0.conj(0);
            } else if (optype[seq]=='n') {
                newblock.Ham.prod(site.op[seq][0], tmp, basis, 1.0, 'n', 'n');
            }
        }
    }

    cout << endl << "newHammem=" << newblock.Ham.mem_size()/1024 << "GB ---> ";
    newblock.Ham.trunc(trunc);
    // newblock.Ham=newblock.Ham.applytrunc(trunc);
    cout << newblock.Ham.mem_size()/1024 << "GB; " << endl;

    // update operators
    if (sore=='s'){
        for (int seq=0; seq<num_op; ++seq) {
            for (size_t i = 0; i < mybase[seq][block.len()+1].sys_idx.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][block.len()].sys_idx.begin(), mybase[seq][block.len()].sys_idx.end(), mybase[seq][block.len()+1].sys_idx[i]);
                if (iter==mybase[seq][block.len()].sys_idx.end()) {
                    newblock.op[seq][i].prod_id(block.Ham, site.op[seq][0], basis, 1.0, 'l', 0);
                } else {
                    int loc=distance(mybase[seq][block.len()].sys_idx.begin(), iter);
                    newblock.op[seq][i].prod_id(block.op[seq][loc], site.Ham, basis, 1.0, 'r', 0);
                }
                newblock.op[seq][i].trunc(trunc);
                // newblock.op[seq][i]=newblock.op[seq][i].applytrunc(trunc);
            }

        }
    } else {
        for (int seq=0; seq<num_op; ++seq) {
            for (size_t i = 0; i < mybase[seq][ltot-2-(block.len()+1)].env_idx.size(); ++i) {
                vector<int>::iterator iter=find(mybase[seq][ltot-2-block.len()].env_idx.begin(), mybase[seq][ltot-2-block.len()].env_idx.end(), mybase[seq][ltot-2-(block.len()+1)].env_idx[i]);
                if (iter==mybase[seq][ltot-2-block.len()].env_idx.end()) {
                    newblock.op[seq][i].prod_id(site.op[seq][0], block.Ham, basis, 1.0, 'r', 0);
                } else {
                    int loc=distance(mybase[seq][ltot-2-block.len()].env_idx.begin(), iter);
                    newblock.op[seq][i].prod_id(site.Ham, block.op[seq][loc], basis, 1.0, 'l', 0);
                }
                newblock.op[seq][i].trunc(trunc);
                // newblock.op[seq][i]=newblock.op[seq][i].applytrunc(trunc);
            }
            // for (int i=0; i<block.opl[seq]; ++i) {
            //     block.op[seq][i].clear();
            // }
            // delete [] block.op[seq]; block.op[seq]=NULL;
        }
    }

    block=newblock;

}

void Htowave(const Hamilton &sys, const wave &trail, wave &newwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, cudaStream_t stream) {
    // wave newwave(trail, stream);
    newwave.setzero(stream);

    //-----------------
    for (int seq=0; seq<num_op; ++seq) {
        if (sys.len()+env.len()+2==ltot) {
            if (mybase[seq][sys.len()].sys_st1.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].sys_st1.size(); ++i) {
                    vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[i].l1);
                    int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                    tmp.mul_add( mybase[seq][sys.len()].sys_st1[i].amp[seq], sys.op[seq][l1], stream);
                }
                if (optype[seq]=='c') {
                    newwave.mul(tmp, siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, 1.0, "tnii", sys_basis, env_basis, stream);
                    newwave.mul(tmp, siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*1.0, "ntii", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(tmp, siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, 1.0, "nnii", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].sys_st1.size()==1) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[0].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                if (optype[seq]=='c') {
                    newwave.mul(sys.op[seq][l1], siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[0].amp[seq], "tnii", sys_basis, env_basis, stream);
                    newwave.mul(sys.op[seq][l1], siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st1[0].amp[seq], "ntii", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.op[seq][l1], siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[0].amp[seq], "nnii", sys_basis, env_basis, stream);
                }
            }

            if (mybase[seq][sys.len()].st2_env.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].st2_env.size(); ++i) {
                    vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st2_env[i].l2);
                    int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                    tmp.mul_add( mybase[seq][sys.len()].st2_env[i].amp[seq], env.op[seq][l2], stream);
                }
                if (optype[seq]=='c') {
                    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], tmp, 1.0, "iitn", sys_basis, env_basis, stream);
                    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], tmp, site.op[seq][0].sign()*1.0, "iint", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], tmp, 1.0, "iinn", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].st2_env.size()==1) {
                vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st2_env[0].l2);
                int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                if (optype[seq]=='c') {
                    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][sys.len()].st2_env[0].amp[seq], "iitn", sys_basis, env_basis, stream);
                    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][sys.len()].st2_env[0].amp[seq], "iint", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][sys.len()].st2_env[0].amp[seq], "iinn", sys_basis, env_basis, stream);
                }
            }

            if (mybase[seq][sys.len()].sys_st2.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].sys_st2.size(); ++i) {
                    vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st2[i].l1);
                    int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                    tmp.mul_add( mybase[seq][sys.len()].sys_st2[i].amp[seq], sys.op[seq][l1], stream);
                }
                if (optype[seq]=='c') {
                    newwave.mul(tmp, siteCPU.Ham, trail, siteCPU.op[seq][0], env.Ham, 1.0, "tini", sys_basis, env_basis, stream);
                    newwave.mul(tmp, siteCPU.Ham, trail, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*1.0, "niti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(tmp, siteCPU.Ham, trail, siteCPU.op[seq][0], env.Ham, 1.0, "nini", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].sys_st2.size()==1) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st2[0].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                if (optype[seq]=='c') {
                    newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].sys_st2[0].amp[seq], "tini", sys_basis, env_basis, stream);
                    newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st2[0].amp[seq], "niti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].sys_st2[0].amp[seq], "nini", sys_basis, env_basis, stream);
                }
            }
            
            if (mybase[seq][sys.len()].st1_env.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i=0; i<mybase[seq][sys.len()].st1_env.size(); ++i) {
                    vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st1_env[i].l2);
                    int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                    tmp.mul_add( mybase[seq][sys.len()].st1_env[i].amp[seq], env.op[seq][l2], stream);
                }
                if (optype[seq]=='c') {
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.Ham, tmp, 1.0, "itin", sys_basis, env_basis, stream);
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.Ham, tmp, site.op[seq][0].sign()*1.0, "init", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.Ham, tmp, 1.0, "inin", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].st1_env.size()==1) {
                vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st1_env[0].l2);
                int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                if (optype[seq]=='c') {
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].st1_env[0].amp[seq], "itin", sys_basis, env_basis, stream);
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.Ham, env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][sys.len()].st1_env[0].amp[seq], "init", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].st1_env[0].amp[seq], "inin", sys_basis, env_basis, stream);
                }
            }
            
            if ( mybase[seq][sys.len()].st1_st2.size()>0) {
                assert(mybase[seq][sys.len()].st1_st2.size()==1);
                if (optype[seq]=='c') {
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].st1_st2[0].amp[seq], "itni", sys_basis, env_basis, stream);
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].st1_st2[0].amp[seq], "inti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    newwave.mul(sys.Ham, siteCPU.op[seq][0], trail, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].st1_st2[0].amp[seq], "inni", sys_basis, env_basis, stream);
                }
                
            }

            if (mybase[seq][sys.len()].sys_env.size()>0) {
                vector<int> count;
                count=sys_env_helper(mybase[seq][sys.len()].sys_env);
                vector<int>::iterator itersys, iterenv;
                int l1=0, l2=0;
                
                for (size_t i=1; i<=mybase[seq][sys.len()].sys_env.size(); ++i) {
                    reducematrix tmp(0, env.op[seq][0].sign());
                    bool myflag=false;
                    for (size_t j=0; j<count.size(); ++j) {
                        if (count[j]==i) {
                            itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                            l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                            iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                            l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                            tmp.mul_add( mybase[seq][sys.len()].sys_env[j].amp[seq], env.op[seq][l2], stream);
                            myflag=true;
                        }
                    }
                    if (myflag) {
                        if (optype[seq]=='c') {
                            newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.Ham, tmp, 1.0, "tiin", sys_basis, env_basis, stream);
                            newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.Ham, tmp, sys.op[seq][l1].sign()*1.0, "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.Ham, tmp, 1.0, "niin", sys_basis, env_basis, stream);
                        }
                    }
                }
                for (int i= - int(mybase[seq][sys.len()].sys_env.size()); i<0; ++i) {
                    reducematrix tmp(0, sys.op[seq][l1].sign());
                    bool myflag=false;
                    for (size_t j=0; j<count.size(); ++j) {
                        if (count[j]==i) {
                            itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                            l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                            iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                            l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                            tmp.mul_add( mybase[seq][sys.len()].sys_env[j].amp[seq], sys.op[seq][l1], stream);
                            myflag=true;
                        }
                    }
                    if (myflag) {
                        if (optype[seq]=='c') {
                            newwave.mul(tmp, siteCPU.Ham, trail, siteCPU.Ham, env.op[seq][l2], 1.0, "tiin", sys_basis, env_basis, stream);
                            newwave.mul(tmp, siteCPU.Ham, trail, siteCPU.Ham, env.op[seq][l2], env.op[seq][l2].sign()*1.0, "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            newwave.mul(tmp, siteCPU.Ham, trail, siteCPU.Ham, env.op[seq][l2], 1.0, "niin", sys_basis, env_basis, stream);
                        }
                    }
                }
                
                for (size_t j=0; j<count.size(); ++j) {
                    if (count[j]==0) {
                        itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                        l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                        iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                        l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                        if (optype[seq]=='c') {
                            newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].sys_env[j].amp[seq], "tiin", sys_basis, env_basis, stream);
                            newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.Ham, env.op[seq][l2], sys.op[seq][l1].sign()*mybase[seq][sys.len()].sys_env[j].amp[seq], "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            newwave.mul(sys.op[seq][l1], siteCPU.Ham, trail, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].sys_env[j].amp[seq], "niin", sys_basis, env_basis, stream);
                        }
                        
                    }
                }
            }

        } else {
            for (size_t i = 0; i < mybase[seq][sys.len()].sys_st1.size(); ++i) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[i].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);

                if (mybase[seq][sys.len()].sys_st1[i].amp[seq] != 0) {
                    if (optype[seq]=='c') {
                        newwave.mul(sys.op[seq][l1], siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[i].amp[seq], "tnii", sys_basis, env_basis, stream);
                        newwave.mul(sys.op[seq][l1], siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st1[i].amp[seq], "ntii", sys_basis, env_basis, stream);
                    } else if (optype[seq]=='n') {
                        newwave.mul(sys.op[seq][l1], siteCPU.op[seq][0], trail, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[i].amp[seq], "nnii", sys_basis, env_basis, stream);
                    }
                }
            }
            for (size_t i = 0; i < mybase[seq][ltot-2-env.len()].st2_env.size(); ++i) {
                vector<int>::iterator iterenv=find(mybase[seq][ltot-2-env.len()].env_idx.begin(), mybase[seq][ltot-2-env.len()].env_idx.end(), mybase[seq][ltot-2-env.len()].st2_env[i].l2);
                int l2=distance(mybase[seq][ltot-2-env.len()].env_idx.begin(), iterenv);
                if (mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq] != 0) {
                    if (optype[seq]=='c') {
                        newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iint", sys_basis, env_basis, stream);
                        newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iitn", sys_basis, env_basis, stream);
                    } else if (optype[seq]=='n') {
                        newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iinn", sys_basis, env_basis, stream);
                    }
                }
            }
        }
    }

    HamiltonCPU msitCPU;
    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.Ham, env.Ham, 1.0, "niii", sys_basis, env_basis, stream);
    train_site[sys.len()+1].toCPU(msitCPU);
    newwave.mul(sys.Ham, msitCPU.Ham, trail, siteCPU.Ham, env.Ham, 1.0, "inii", sys_basis, env_basis, stream);
    train_site[ltot-env.len()].toCPU(msitCPU);
    newwave.mul(sys.Ham, siteCPU.Ham, trail, msitCPU.Ham, env.Ham, 1.0, "iini", sys_basis, env_basis, stream);
    newwave.mul(sys.Ham, siteCPU.Ham, trail, siteCPU.Ham, env.Ham, 1.0, "iiin", sys_basis, env_basis, stream);
    // cout << newwave << endl;

    // return newwave;
}

void HtowaveCtoG(const Hamilton &sys, const wave_CPU &trail, wave &newwave, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, cudaStream_t stream) {
    // wave newwave(trail, stream);
    newwave.setzero(stream);

    //-----------------
    for (int seq=0; seq<num_op; ++seq) {
        if (sys.len()+env.len()+2==ltot) {
            if (mybase[seq][sys.len()].sys_st1.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].sys_st1.size(); ++i) {
                    vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[i].l1);
                    int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                    tmp.mul_add( mybase[seq][sys.len()].sys_st1[i].amp[seq], sys.op[seq][l1], stream);
                }
                if (optype[seq]=='c') {
                    mul_CtoG(tmp, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, 1.0, "tnii", sys_basis, env_basis, stream);
                    mul_CtoG(tmp, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*1.0, "ntii", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(tmp, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, 1.0, "nnii", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].sys_st1.size()==1) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[0].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                if (optype[seq]=='c') {
                    mul_CtoG(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[0].amp[seq], "tnii", sys_basis, env_basis, stream);
                    mul_CtoG(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st1[0].amp[seq], "ntii", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[0].amp[seq], "nnii", sys_basis, env_basis, stream);
                }
            }

            if (mybase[seq][sys.len()].st2_env.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].st2_env.size(); ++i) {
                    vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st2_env[i].l2);
                    int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                    tmp.mul_add( mybase[seq][sys.len()].st2_env[i].amp[seq], env.op[seq][l2], stream);
                }
                if (optype[seq]=='c') {
                    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], tmp, 1.0, "iitn", sys_basis, env_basis, stream);
                    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], tmp, site.op[seq][0].sign()*1.0, "iint", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], tmp, 1.0, "iinn", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].st2_env.size()==1) {
                vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st2_env[0].l2);
                int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                if (optype[seq]=='c') {
                    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][sys.len()].st2_env[0].amp[seq], "iitn", sys_basis, env_basis, stream);
                    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][sys.len()].st2_env[0].amp[seq], "iint", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][sys.len()].st2_env[0].amp[seq], "iinn", sys_basis, env_basis, stream);
                }
            }

            if (mybase[seq][sys.len()].sys_st2.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].sys_st2.size(); ++i) {
                    vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st2[i].l1);
                    int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                    tmp.mul_add( mybase[seq][sys.len()].sys_st2[i].amp[seq], sys.op[seq][l1], stream);
                }
                if (optype[seq]=='c') {
                    mul_CtoG(tmp, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.Ham, 1.0, "tini", sys_basis, env_basis, stream);
                    mul_CtoG(tmp, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*1.0, "niti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(tmp, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.Ham, 1.0, "nini", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].sys_st2.size()==1) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st2[0].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                if (optype[seq]=='c') {
                    mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].sys_st2[0].amp[seq], "tini", sys_basis, env_basis, stream);
                    mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st2[0].amp[seq], "niti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].sys_st2[0].amp[seq], "nini", sys_basis, env_basis, stream);
                }
            }
            
            if (mybase[seq][sys.len()].st1_env.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i=0; i<mybase[seq][sys.len()].st1_env.size(); ++i) {
                    vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st1_env[i].l2);
                    int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                    tmp.mul_add( mybase[seq][sys.len()].st1_env[i].amp[seq], env.op[seq][l2], stream);
                }
                if (optype[seq]=='c') {
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, tmp, 1.0, "itin", sys_basis, env_basis, stream);
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, tmp, site.op[seq][0].sign()*1.0, "init", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, tmp, 1.0, "inin", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].st1_env.size()==1) {
                vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st1_env[0].l2);
                int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                if (optype[seq]=='c') {
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].st1_env[0].amp[seq], "itin", sys_basis, env_basis, stream);
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][sys.len()].st1_env[0].amp[seq], "init", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].st1_env[0].amp[seq], "inin", sys_basis, env_basis, stream);
                }
            }
            
            if ( mybase[seq][sys.len()].st1_st2.size()>0) {
                assert(mybase[seq][sys.len()].st1_st2.size()==1);
                if (optype[seq]=='c') {
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].st1_st2[0].amp[seq], "itni", sys_basis, env_basis, stream);
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].st1_st2[0].amp[seq], "inti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_CtoG(sys.Ham, siteCPU.op[seq][0], trail, newwave, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].st1_st2[0].amp[seq], "inni", sys_basis, env_basis, stream);
                }
                
            }

            if (mybase[seq][sys.len()].sys_env.size()>0) {
                vector<int> count;
                count=sys_env_helper(mybase[seq][sys.len()].sys_env);
                vector<int>::iterator itersys, iterenv;
                int l1=0, l2=0;
                
                for (size_t i=1; i<=mybase[seq][sys.len()].sys_env.size(); ++i) {
                    reducematrix tmp(0, env.op[seq][0].sign());
                    bool myflag=false;
                    for (size_t j=0; j<count.size(); ++j) {
                        if (count[j]==i) {
                            itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                            l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                            iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                            l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                            tmp.mul_add( mybase[seq][sys.len()].sys_env[j].amp[seq], env.op[seq][l2], stream);
                            myflag=true;
                        }
                    }
                    if (myflag) {
                        if (optype[seq]=='c') {
                            mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.Ham, tmp, 1.0, "tiin", sys_basis, env_basis, stream);
                            mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.Ham, tmp, sys.op[seq][l1].sign()*1.0, "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.Ham, tmp, 1.0, "niin", sys_basis, env_basis, stream);
                        }
                    }
                }
                for (int i= - int(mybase[seq][sys.len()].sys_env.size()); i<0; ++i) {
                    reducematrix tmp(0, sys.op[seq][l1].sign());
                    bool myflag=false;
                    for (size_t j=0; j<count.size(); ++j) {
                        if (count[j]==i) {
                            itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                            l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                            iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                            l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                            tmp.mul_add( mybase[seq][sys.len()].sys_env[j].amp[seq], sys.op[seq][l1], stream);
                            myflag=true;
                        }
                    }
                    if (myflag) {
                        if (optype[seq]=='c') {
                            mul_CtoG(tmp, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.op[seq][l2], 1.0, "tiin", sys_basis, env_basis, stream);
                            mul_CtoG(tmp, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.op[seq][l2], env.op[seq][l2].sign()*1.0, "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            mul_CtoG(tmp, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.op[seq][l2], 1.0, "niin", sys_basis, env_basis, stream);
                        }
                    }
                }
                
                for (size_t j=0; j<count.size(); ++j) {
                    if (count[j]==0) {
                        itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                        l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                        iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                        l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                        if (optype[seq]=='c') {
                            mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].sys_env[j].amp[seq], "tiin", sys_basis, env_basis, stream);
                            mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.Ham, env.op[seq][l2], sys.op[seq][l1].sign()*mybase[seq][sys.len()].sys_env[j].amp[seq], "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            mul_CtoG(sys.op[seq][l1], siteCPU.Ham, trail, newwave, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].sys_env[j].amp[seq], "niin", sys_basis, env_basis, stream);
                        }
                        
                    }
                }
            }

        } else {
            for (size_t i = 0; i < mybase[seq][sys.len()].sys_st1.size(); ++i) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[i].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);

                if (mybase[seq][sys.len()].sys_st1[i].amp[seq] != 0) {
                    if (optype[seq]=='c') {
                        mul_CtoG(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[i].amp[seq], "tnii", sys_basis, env_basis, stream);
                        mul_CtoG(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st1[i].amp[seq], "ntii", sys_basis, env_basis, stream);
                    } else if (optype[seq]=='n') {
                        mul_CtoG(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwave, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[i].amp[seq], "nnii", sys_basis, env_basis, stream);
                    }
                }
            }
            for (size_t i = 0; i < mybase[seq][ltot-2-env.len()].st2_env.size(); ++i) {
                vector<int>::iterator iterenv=find(mybase[seq][ltot-2-env.len()].env_idx.begin(), mybase[seq][ltot-2-env.len()].env_idx.end(), mybase[seq][ltot-2-env.len()].st2_env[i].l2);
                int l2=distance(mybase[seq][ltot-2-env.len()].env_idx.begin(), iterenv);
                if (mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq] != 0) {
                    if (optype[seq]=='c') {
                        mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iint", sys_basis, env_basis, stream);
                        mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iitn", sys_basis, env_basis, stream);
                    } else if (optype[seq]=='n') {
                        mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iinn", sys_basis, env_basis, stream);
                    }
                }
            }
        }
    }

    // Hamilton msit;
    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.Ham, 1.0, "niii", sys_basis, env_basis, stream);
    // msit=define_site(train[sys.len()+1]);
    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.Ham, 1.0, "inii", sys_basis, env_basis, stream);
	// msit=define_site(train[ltot-env.len()]);
    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.Ham, 1.0, "iini", sys_basis, env_basis, stream);
    mul_CtoG(sys.Ham, siteCPU.Ham, trail, newwave, siteCPU.Ham, env.Ham, 1.0, "iiin", sys_basis, env_basis, stream);
    // cout << newwave << endl;

    // return newwave;
}

void HtowaveGtoC(const Hamilton &sys, const wave &trail, wave_CPU &newwaveCPU, const Hamilton &env, const vector<repmap> &sys_basis, const vector<repmap> &env_basis, cudaStream_t stream) {
    for (int seq=0; seq<num_op; ++seq) {
        if (sys.len()+env.len()+2==ltot) {
            if (mybase[seq][sys.len()].sys_st1.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].sys_st1.size(); ++i) {
                    vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[i].l1);
                    int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                    tmp.mul_add( mybase[seq][sys.len()].sys_st1[i].amp[seq], sys.op[seq][l1], stream);
                }
                if (optype[seq]=='c') {
                    mul_GtoC(tmp, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, 1.0, "tnii", sys_basis, env_basis, stream);
                    mul_GtoC(tmp, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*1.0, "ntii", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(tmp, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, 1.0, "nnii", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].sys_st1.size()==1) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[0].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                if (optype[seq]=='c') {
                    mul_GtoC(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[0].amp[seq], "tnii", sys_basis, env_basis, stream);
                    mul_GtoC(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st1[0].amp[seq], "ntii", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[0].amp[seq], "nnii", sys_basis, env_basis, stream);
                }
            }

            if (mybase[seq][sys.len()].st2_env.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].st2_env.size(); ++i) {
                    vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st2_env[i].l2);
                    int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                    tmp.mul_add( mybase[seq][sys.len()].st2_env[i].amp[seq], env.op[seq][l2], stream);
                }
                if (optype[seq]=='c') {
                    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], tmp, 1.0, "iitn", sys_basis, env_basis, stream);
                    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], tmp, site.op[seq][0].sign()*1.0, "iint", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], tmp, 1.0, "iinn", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].st2_env.size()==1) {
                vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st2_env[0].l2);
                int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                if (optype[seq]=='c') {
                    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][sys.len()].st2_env[0].amp[seq], "iitn", sys_basis, env_basis, stream);
                    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][sys.len()].st2_env[0].amp[seq], "iint", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][sys.len()].st2_env[0].amp[seq], "iinn", sys_basis, env_basis, stream);
                }
            }

            if (mybase[seq][sys.len()].sys_st2.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i = 0; i < mybase[seq][sys.len()].sys_st2.size(); ++i) {
                    vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st2[i].l1);
                    int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                    tmp.mul_add( mybase[seq][sys.len()].sys_st2[i].amp[seq], sys.op[seq][l1], stream);
                }
                if (optype[seq]=='c') {
                    mul_GtoC(tmp, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, 1.0, "tini", sys_basis, env_basis, stream);
                    mul_GtoC(tmp, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*1.0, "niti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(tmp, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, 1.0, "nini", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].sys_st2.size()==1) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st2[0].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                if (optype[seq]=='c') {
                    mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].sys_st2[0].amp[seq], "tini", sys_basis, env_basis, stream);
                    mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st2[0].amp[seq], "niti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].sys_st2[0].amp[seq], "nini", sys_basis, env_basis, stream);
                }
            }
            
            if (mybase[seq][sys.len()].st1_env.size()>1) {
                reducematrix tmp(0, site.op[seq][0].sign());
                for (size_t i=0; i<mybase[seq][sys.len()].st1_env.size(); ++i) {
                    vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st1_env[i].l2);
                    int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                    tmp.mul_add( mybase[seq][sys.len()].st1_env[i].amp[seq], env.op[seq][l2], stream);
                }
                if (optype[seq]=='c') {
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, tmp, 1.0, "itin", sys_basis, env_basis, stream);
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, tmp, site.op[seq][0].sign()*1.0, "init", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, tmp, 1.0, "inin", sys_basis, env_basis, stream);
                }
            } else if (mybase[seq][sys.len()].st1_env.size()==1) {
                vector<int>::iterator iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].st1_env[0].l2);
                int l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                if (optype[seq]=='c') {
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].st1_env[0].amp[seq], "itin", sys_basis, env_basis, stream);
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][sys.len()].st1_env[0].amp[seq], "init", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].st1_env[0].amp[seq], "inin", sys_basis, env_basis, stream);
                }
            }
            
            if ( mybase[seq][sys.len()].st1_st2.size()>0) {
                assert(mybase[seq][sys.len()].st1_st2.size()==1);
                if (optype[seq]=='c') {
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].st1_st2[0].amp[seq], "itni", sys_basis, env_basis, stream);
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].st1_st2[0].amp[seq], "inti", sys_basis, env_basis, stream);
                } else if (optype[seq]=='n') {
                    mul_GtoC(sys.Ham, siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.op[seq][0], env.Ham, mybase[seq][sys.len()].st1_st2[0].amp[seq], "inni", sys_basis, env_basis, stream);
                }
                
            }

            if (mybase[seq][sys.len()].sys_env.size()>0) {
                vector<int> count;
                count=sys_env_helper(mybase[seq][sys.len()].sys_env);
                vector<int>::iterator itersys, iterenv;
                int l1=0, l2=0;
                
                for (size_t i=1; i<=mybase[seq][sys.len()].sys_env.size(); ++i) {
                    reducematrix tmp(0, env.op[seq][0].sign());
                    bool myflag=false;
                    for (size_t j=0; j<count.size(); ++j) {
                        if (count[j]==i) {
                            itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                            l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                            iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                            l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                            tmp.mul_add( mybase[seq][sys.len()].sys_env[j].amp[seq], env.op[seq][l2], stream);
                            myflag=true;
                        }
                    }
                    if (myflag) {
                        if (optype[seq]=='c') {
                            mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, tmp, 1.0, "tiin", sys_basis, env_basis, stream);
                            mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, tmp, sys.op[seq][l1].sign()*1.0, "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, tmp, 1.0, "niin", sys_basis, env_basis, stream);
                        }
                    }
                }
                for (int i= - int(mybase[seq][sys.len()].sys_env.size()); i<0; ++i) {
                    reducematrix tmp(0, sys.op[seq][l1].sign());
                    bool myflag=false;
                    for (size_t j=0; j<count.size(); ++j) {
                        if (count[j]==i) {
                            itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                            l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                            iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                            l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                            tmp.mul_add( mybase[seq][sys.len()].sys_env[j].amp[seq], sys.op[seq][l1], stream);
                            myflag=true;
                        }
                    }
                    if (myflag) {
                        if (optype[seq]=='c') {
                            mul_GtoC(tmp, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], 1.0, "tiin", sys_basis, env_basis, stream);
                            mul_GtoC(tmp, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], env.op[seq][l2].sign()*1.0, "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            mul_GtoC(tmp, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], 1.0, "niin", sys_basis, env_basis, stream);
                        }
                    }
                }
                
                for (size_t j=0; j<count.size(); ++j) {
                    if (count[j]==0) {
                        itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_env[j].l1);
                        l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);
                        iterenv=find(mybase[seq][sys.len()].env_idx.begin(), mybase[seq][sys.len()].env_idx.end(), mybase[seq][sys.len()].sys_env[j].l2);
                        l2=distance(mybase[seq][sys.len()].env_idx.begin(), iterenv);
                        if (optype[seq]=='c') {
                            mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].sys_env[j].amp[seq], "tiin", sys_basis, env_basis, stream);
                            mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], sys.op[seq][l1].sign()*mybase[seq][sys.len()].sys_env[j].amp[seq], "niit", sys_basis, env_basis, stream);
                        } else if (optype[seq]=='n') {
                            mul_GtoC(sys.op[seq][l1], siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.op[seq][l2], mybase[seq][sys.len()].sys_env[j].amp[seq], "niin", sys_basis, env_basis, stream);
                        }
                        
                    }
                }
            }
        } else {
            for (size_t i = 0; i < mybase[seq][sys.len()].sys_st1.size(); ++i) {
                vector<int>::iterator itersys=find(mybase[seq][sys.len()].sys_idx.begin(), mybase[seq][sys.len()].sys_idx.end(), mybase[seq][sys.len()].sys_st1[i].l1);
                int l1=distance(mybase[seq][sys.len()].sys_idx.begin(), itersys);

                if (mybase[seq][sys.len()].sys_st1[i].amp[seq] != 0) {
                    if (optype[seq]=='c') {
                        mul_GtoC(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[i].amp[seq], "tnii", sys_basis, env_basis, stream);
                        mul_GtoC(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, site.op[seq][0].sign()*mybase[seq][sys.len()].sys_st1[i].amp[seq], "ntii", sys_basis, env_basis, stream);
                    } else if (optype[seq]=='n') {
                        mul_GtoC(sys.op[seq][l1], siteCPU.op[seq][0], trail, newwaveCPU, siteCPU.Ham, env.Ham, mybase[seq][sys.len()].sys_st1[i].amp[seq], "nnii", sys_basis, env_basis, stream);
                    }
                }
            }
            for (size_t i = 0; i < mybase[seq][ltot-2-env.len()].st2_env.size(); ++i) {
                vector<int>::iterator iterenv=find(mybase[seq][ltot-2-env.len()].env_idx.begin(), mybase[seq][ltot-2-env.len()].env_idx.end(), mybase[seq][ltot-2-env.len()].st2_env[i].l2);
                int l2=distance(mybase[seq][ltot-2-env.len()].env_idx.begin(), iterenv);
                if (mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq] != 0) {
                    if (optype[seq]=='c') {
                        mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iint", sys_basis, env_basis, stream);
                        mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.op[seq][l2], site.op[seq][0].sign()*mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iitn", sys_basis, env_basis, stream);
                    } else if (optype[seq]=='n') {
                        mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.op[seq][0], env.op[seq][l2], mybase[seq][ltot-2-env.len()].st2_env[i].amp[seq], "iinn", sys_basis, env_basis, stream);
                    }
                }
            }
        }
    }

    Hamilton msit;
    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.Ham, 1.0, "niii", sys_basis, env_basis, stream);
    // msit=define_site(train[sys.len()+1]);
    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.Ham, 1.0, "inii", sys_basis, env_basis, stream);
    // msit=define_site(train[ltot-env.len()]);
    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.Ham, 1.0, "iini", sys_basis, env_basis, stream);
    mul_GtoC(sys.Ham, siteCPU.Ham, trail, newwaveCPU, siteCPU.Ham, env.Ham, 1.0, "iiin", sys_basis, env_basis, stream);

}