#include "measure.hpp"

double singlesite(const reducematrix &myop, const int &siteid, cudaStream_t stream[2]);
double twosite(const reducematrix &op1, const reducematrix &op2, const int &opid1, const int &opid2, cudaStream_t stream[2]);

// will be replaced by op_list soon
namespace spinless {
    void measurehelp(cudaStream_t stream[2]) {
        // measuresite(fn, "out/n.dat", stream);
        // measurecorr_quick(fc_u_d, fc_u, 1.0, "out/fcor.dat", stream);

        int o_size=1;
        reducematrix* opl1=new reducematrix[o_size] ();
        reducematrix* opl2=new reducematrix[o_size] ();
        opl1[0]=fc_u_d; opl2[0]=fc_u;
        measurecorr_quick_list(opl1, opl2, o_size, 1.0, "out/fcor.dat", stream);
        for (int i_o = 0; i_o < o_size; i_o++) {
            opl1[i_o].clear(); opl2[i_o].clear();
        }
        delete [] opl1; delete [] opl2;

        measuresccor_spinless_quick("out/SCcor.dat", stream);
    }
}

namespace Hubbard {
    void measurehelp(cudaStream_t stream[2]) {
        // measuresite(fn, "out/n.dat", stream);
        // measuresite(s_z, "out/Sz.dat", stream);

        // measurecorr_quick(s_z, s_z, 1.0, "out/SzCor.dat", stream);
        // measurecorr_quick(fn, fn, 1.0, "out/fnCor.dat", stream);
        measuresccor_quick("out/FourSC.dat", stream);// to be developed
        // measurecorr_quick(fc_u_d, fc_u, 1.0, "out/fcupCor.dat", stream);

        int o_size=3;
        reducematrix* opl1=new reducematrix[o_size] ();
        reducematrix* opl2=new reducematrix[o_size] ();
        opl1[0]=s_z; opl1[1]=fc_u_d; opl1[2]=fn; 
        opl2[0]=s_z; opl2[1]=fc_u; opl2[2]=fn;
        measurecorr_quick_list(opl1, opl2, o_size, 1.0, "out/Cor_Sz_fc_fn.dat", stream);
        for (int i_o = 0; i_o < o_size; i_o++) {
            opl1[i_o].clear(); opl2[i_o].clear();
        }
        delete [] opl1; delete [] opl2;

    }
}

namespace Hubbard_bond {
    void measurehelp(cudaStream_t stream[2]) {
        // measuresite(fn, "out/n.dat", stream);
        // measuresite(s_z, "out/Sz.dat", stream);

        // measurecorr_quick(s_z, s_z, 1.0, "out/SzCor.dat", stream);
        // measurecorr_quick(fn, fn, 1.0, "out/fnCor.dat", stream);
        // measuresccor_quick("out/FourSC.dat", stream);// to be developed
        // measurecorr_quick(fc_u_d, fc_u, 1.0, "out/fcupCor.dat", stream);

        measurebond(fc_u_d, fc_u, 1.0, "out/bonds.dat", stream);

        int o_size=3;
        reducematrix* opl1=new reducematrix[o_size] ();
        reducematrix* opl2=new reducematrix[o_size] ();
        opl1[0]=s_z; opl1[1]=fc_u_d; opl1[2]=fn;
        opl2[0]=s_z; opl2[1]=fc_u; opl2[2]=fn;
        measurecorr_quick_list(opl1, opl2, o_size, 1.0, "out/Cor_Sz_fc_fn.dat", stream);
        for (int i_o = 0; i_o < o_size; i_o++) {
            opl1[i_o].clear(); opl2[i_o].clear();
        }
        delete [] opl1; delete [] opl2;
    }

    void measurebond(const reducematrix &op1, const reducematrix &op2, const double &para, const string &name, cudaStream_t stream[2]) {
        ofstream out(name.c_str(), ios::out | ios::trunc);
        if (out.is_open()) {
            out << scientific;
            double myval;
            for (int i = 0; i < allbonds.size(); ++i) {
                myval=twosite(op1,op2,allbonds[i].l1+1,allbonds[i].l2+1, stream);
                out << allbonds[i].l1+1 << " " << allbonds[i].l2+1 << " " << para*myval << endl;
            }
            out << endl;
            out.close();
        }   
    }
}

namespace tUJmodel {
    void measurehelp(cudaStream_t stream[2]) {
        // measuresite(fn, "out/n.dat", stream);
        // measuresite(s_z, "out/Sz.dat", stream);

        // measurecorr_quick(s_z, s_z, 1.0, "out/SzCor.dat", stream);
        // measurecorr_quick(fn, fn, 1.0, "out/fnCor.dat", stream);
        measuresccor_quick("out/FourSC.dat", stream);// to be developed
        // measurecorr_quick(fc_u_d, fc_u, 1.0, "out/fcupCor.dat", stream);

        int o_size=3;
        reducematrix* opl1=new reducematrix[o_size] ();
        reducematrix* opl2=new reducematrix[o_size] ();
        opl1[0]=s_z; opl1[1]=fc_u_d; opl1[2]=fpair;
        opl2[0]=s_z; opl2[1]=fc_u; opl2[2]=fpair.conj(0);
        measurecorr_quick_list(opl1, opl2, o_size, 1.0, "out/Cor_Sz_fc_fpair.dat", stream);
        for (int i_o = 0; i_o < o_size; i_o++) {
            opl1[i_o].clear(); opl2[i_o].clear();
        }
        delete [] opl1; delete [] opl2;

    }
}

namespace tJmodel {
    void measurehelp(cudaStream_t stream[2]) {
        // measuresite(fn, "out/n.dat", stream);
        // measuresite(s_z, "out/Sz.dat", stream);
        // measurecorr_quick(s_z, s_z, 1.0, "out/SzCor.dat", stream);
        // measurecorr_quick(fc_u_d, fc_u, 1.0, "out/fcupCor.dat", stream);

        int o_size=2;
        reducematrix* opl1=new reducematrix[o_size] ();
        reducematrix* opl2=new reducematrix[o_size] ();
        opl1[0]=s_z; opl1[1]=fc_u_d;
        opl2[0]=s_z; opl2[1]=fc_u;
        measurecorr_quick_list(opl1, opl2, o_size, 1.0, "out/Cor_Sz_fc.dat", stream);
        for (int i_o = 0; i_o < o_size; i_o++) {
            opl1[i_o].clear(); opl2[i_o].clear();
        }
        delete [] opl1; delete [] opl2;

        measuresccor_quick("out/FourSC.dat", stream);
    }
}

namespace Heisenberg {
    void measurehelp(cudaStream_t stream[2]) {
        // measuresite(s_z, "out/Sz.dat", stream);
        measurecorr_quick(s_z, s_z, 1.0, "out/SzCor.dat", stream);
    }
}

void measuresite(const reducematrix &op, const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific;
        for (int i = 1; i <= ltot; ++i) {
            out << i << " " << singlesite(op,i, stream) << endl;
        }
        out << endl;
        out.close();
    }
}

void measurecorr(const reducematrix &op1, const reducematrix &op2, const double &para, const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        out << "ref = " << twopoint[0] << endl;
        for (int i = 1; i < twopoint.size(); ++i) {
                out << i << " " << para*twosite(op1,op2,twopoint[0],twopoint[i], stream) << endl;
        }
        out << endl;
        out.close();
    }
}

void read_sys_trun_help(reducematrix &trun, const int &i, const int &endid, cudaStream_t stream) {
    if (i>truncpoint && i<endid) {
        trun.fromdisk(file + "systrun" + to_string(i), 'n', stream);
    }
}

void move_sysop(reducematrix &op, const int &begid, const int &endid, const bool &create, cudaStream_t stream[2]) {
	reducematrix newop(0, op.sign());
	vector<repmap> basis;
	if (create) {
		if (begid > 1) {
            mapfromdisk(basis, file + "sysmap" + to_string(begid));
            reducematrix id1 = basistoid(basis, 'l');
			newop.prod_id(id1, op, basis, 1.0, 'l', stream[0]);
		} else {
			basis = jmap(Jtarget.Ham, site.Ham);
			newop.prod_id(Jtarget.Ham, op, basis, 1.0, 'l', stream[0]);
		}
		op = newop;
	}

	// from untrunc to untrunc
	if (begid<endid) {
		if (begid>truncpoint) {
            reducematrix trun;
            trun.fromdisk(file + "systrun" + to_string(begid), 'n', stream[0]);
			op = op.applytrunc(trun, stream[0]);
		}

        reducematrix sys_trun;
		for (int i = begid+1; i < endid; ++i) {
            // int myi=i+1;
            // thread th0(read_sys_trun_help, ref(sys_trun[(myi)%2]), ref(myi), ref(endid), ref(stream[1]));
            if (i>truncpoint) {
                sys_trun.fromdisk(file + "systrun" + to_string(i), 'n', stream[0]);
            }
			mapfromdisk(basis, file+"sysmap"+to_string(i));
			newop.clear();
            newop.prod_id(op, site.Ham, basis, 1.0, 'r', stream[0]);
            
            if (i>truncpoint) {
                op.set(newop.applytrunc(sys_trun, stream[0]), stream[0]);
                sys_trun.clear();
			} else {
				op.set(newop, stream[0]);
			}

            // th0.join();
            // cout << op << endl;
		}
		mapfromdisk(basis, file+"sysmap"+to_string(endid));
		newop.clear();
        newop.prod_id(op, site.Ham, basis, 1.0, 'r', stream[0]);
		op = newop;
	}
}

void move_envop(reducematrix &op, const int &begid, const int &endid, const bool &create, cudaStream_t stream[2]) {
	reducematrix newop(0,op.sign());
	vector<repmap> basis;

	if (create && begid > 1) {
        mapfromdisk(basis, file + "envmap" + to_string(begid));
        reducematrix id2 = basistoid(basis, 'r');
		newop.prod_id(op, id2, basis, 1.0, 'r', stream[0]);
		op = newop;
	}
    
    reducematrix env_trun;
	if (begid<endid) {
		if (begid>truncpoint) {
            env_trun.fromdisk(file + "envtrun" + to_string(begid), 'n', stream[1]);
			op=op.applytrunc(env_trun, stream[0]);
            env_trun.clear();
		}
		for (int i = begid+1; i < endid; ++i) {
            if (i>truncpoint) {
                env_trun.fromdisk(file + "envtrun" + to_string(i), 'n', stream[1]);
            }
			mapfromdisk(basis, file+"envmap"+to_string(i));
			newop.clear();
			newop.prod_id(site.Ham, op, basis, 1.0, 'l', stream[0]);
			if (i>truncpoint) {
				op=newop.applytrunc(env_trun, stream[0]);
                env_trun.clear();
			} else {
				op=newop;
			}
		}
		mapfromdisk(basis, file+"envmap"+to_string(endid));
		newop.clear();
		newop.prod_id(site.Ham, op, basis, 1.0, 'l', stream[0]);
		op = newop;
	}
}

void move_sysop_list(reducematrix* op, const int &o_size, const int &begid, const int &endid, const bool &create, cudaStream_t stream[2]) {
	// reducematrix newop(0, op.sign());
	vector<repmap> basis;
	if (create) {
		if (begid > 1) {
            mapfromdisk(basis, file + "sysmap" + to_string(begid));
            reducematrix id1 = basistoid(basis, 'l');
            for (int i_o = 0; i_o < o_size; i_o++) {
                reducematrix newop(0, op[i_o].sign());
    			newop.prod_id(id1, op[i_o], basis, 1.0, 'l', stream[0]);
                op[i_o]=newop;
            }
		} else {
			basis = jmap(Jtarget.Ham, site.Ham);
            for (int i_o = 0; i_o < o_size; i_o++) {
                reducematrix newop(0, op[i_o].sign());
    			newop.prod_id(Jtarget.Ham, op[i_o], basis, 1.0, 'l', stream[0]);
                op[i_o]=newop;
            }
		}
	}
    
	if (begid<endid) {
        reducematrix sys_trun;
		if (begid>truncpoint) {
            sys_trun.fromdisk(file + "systrun" + to_string(begid), 'n', stream[0]);
            for (int i_o = 0; i_o < o_size; i_o++) {
                op[i_o] = op[i_o].applytrunc(sys_trun, stream[0]);
            }
            sys_trun.clear();
		}
		for (int i = begid+1; i < endid; ++i) {
            if (i>truncpoint) {
                sys_trun.fromdisk(file + "systrun" + to_string(i), 'n', stream[0]);
            }
            mapfromdisk(basis, file+"sysmap"+to_string(i));
            for (int i_o = 0; i_o < o_size; i_o++) {
                reducematrix newop(0, op[i_o].sign());
                newop.prod_id(op[i_o], site.Ham, basis, 1.0, 'r', stream[0]);
                if (i>truncpoint) {
                    op[i_o].set(newop.applytrunc(sys_trun, stream[0]), stream[0]);
                    
                } else {
                    op[i_o].set(newop, stream[0]);
                }
            }
            // th0.join();
            // cout << op << endl;
		}
        
		mapfromdisk(basis, file+"sysmap"+to_string(endid));
		for (int i_o = 0; i_o < o_size; i_o++) {
            reducematrix newop(0, op[i_o].sign());
            newop.prod_id(op[i_o], site.Ham, basis, 1.0, 'r', stream[0]);
		    op[i_o] = newop;
        }
	}
}

double sys_measure(reducematrix &myop) {
    double val;
    wave wav(ground, 0);
    wav.setzero(0);
    wav.mul(myop,ground);
    val=wav.dot(ground);
    return val;
}

double env_measure(reducematrix &myop) {
    double val;
    wave wav(ground, 0);
    wav.setzero(0);
    wav.mul(ground,myop);
    val=wav.dot(ground);
    return val;
}

double sys_env_measure(reducematrix &myop1, reducematrix &myop2) {
    double val;
    wave wav(ground, 0);
    wav.setzero(0);
    wav.mul(myop1,ground,myop2,1.0,'n','n');
    val=wav.dot(ground);
    return val;
}

double singlesite(const reducematrix &myop, const int &siteid, cudaStream_t stream[2]) {
	double val;
	reducematrix op;
	op=myop;
	if (siteid<=stopid) {
		move_sysop(op, siteid, stopid, true, stream);
        val=sys_measure(op);
	} else {
		move_envop(op, ltot-siteid+1, ltot-stopid, true, stream);
        val=env_measure(op);
	}
	return val;
}

void measurecorr_quick(const reducematrix &op1, const reducematrix &op2, const double &para, const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        out << "ref = " << twopoint[0] << endl;
        if (twopoint[0]<=stopid) {
            reducematrix myop1;
            myop1=op1;
            move_sysop(myop1, twopoint[0], twopoint[0], true, stream);
            for (int i = 1; i < twopoint.size(); ++i) {
                if (twopoint[i]<=stopid) {
                    reducematrix myop2, myop;
                    myop2=op2;
                    move_sysop(myop1, twopoint[i-1], twopoint[i], false, stream);
                    move_sysop(myop2, twopoint[i], twopoint[i], true, stream);
                    myop=myop1.mul(myop2);
                    move_sysop(myop, twopoint[i], stopid, false, stream);
                    out << i << " " << para*sys_measure(myop) << endl;
                }
            }
        }
        out << endl;
        out.close();
    }
}

void measurecorr_quick_list(reducematrix* op1, reducematrix* op2, const int o_size, const double &para, const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        out << "ref = " << twopoint[0] << endl;
        if (twopoint[twopoint.size()-1]<=stopid) {
            reducematrix* myop1 = new reducematrix[o_size];
            for (int i_o = 0; i_o < o_size; i_o++) {
                myop1[i_o]=op1[i_o];
            }
            move_sysop_list(myop1, o_size, twopoint[0], twopoint[0], true, stream);
            
            for (int i = 1; i < twopoint.size(); ++i) {
                if (twopoint[i]<=stopid) {
                    reducematrix* myop2 = new reducematrix[o_size];
                    for (int i_o = 0; i_o < o_size; i_o++) {
                        myop2[i_o]=op2[i_o];
                    }
                    move_sysop_list(myop1, o_size, twopoint[i-1], twopoint[i], false, stream);
                    move_sysop_list(myop2, o_size, twopoint[i], twopoint[i], true, stream);
                    reducematrix* myop = new reducematrix[o_size];
                    for (int i_o = 0; i_o < o_size; i_o++) {
                        myop[i_o]=myop1[i_o].mul(myop2[i_o]);
                        myop2[i_o].clear();
                    }
                    delete [] myop2;
                    move_sysop_list(myop, o_size, twopoint[i], stopid, false, stream);
                    out << i << " " ;
                    for (int i_o = 0; i_o < o_size; i_o++) {
                        out << sys_measure(myop[i_o]) << " ";
                        myop[i_o].clear();
                    }
                    out << endl;
                    delete [] myop;
                }
            }
            delete [] myop1;
        } else {
            out << "error: lastsite " << twopoint[twopoint.size()-1] << " is larger than stopid " << stopid << endl;
        }
        out << endl;
        out.close();
    }
}

reducematrix two_sys_site(const reducematrix &op1, const reducematrix &op2, const int &opid1, const int &opid2, cudaStream_t stream[2]) {
	int opid=max(opid1,opid2);
	reducematrix myop1, myop2, myop;
	myop1=op1; myop2=op2;
	move_sysop(myop1, opid1, opid, true, stream);
	move_sysop(myop2, opid2, opid, true, stream);
	myop=myop1.mul(myop2);
	move_sysop(myop, opid, stopid, false, stream);
	return myop;
}

reducematrix three_sys_site(const reducematrix &op1, const reducematrix &op2, const reducematrix &op3, const int &opid1, const int &opid2, const int &opid3, cudaStream_t stream[2]) {
    reducematrix myop, myop_tmp;
    myop=op1; myop_tmp=op2;
    move_sysop(myop, opid1, opid2, true, stream);
    move_sysop(myop_tmp, opid2, opid2, true, stream);
    myop=myop.mul(myop_tmp);
    
    myop_tmp=op3;
    move_sysop(myop, opid2, opid3, false, stream);
    move_sysop(myop_tmp, opid3, opid3, true, stream);
    myop=myop.mul(myop_tmp);
    
    move_sysop(myop, opid3, stopid, false, stream);
    return myop;
}

reducematrix four_sys_site(const reducematrix &op1, const reducematrix &op2, const reducematrix &op3, const reducematrix &op4, const int &opid1, const int &opid2, const int &opid3, const int &opid4, cudaStream_t stream[2]) {
    reducematrix myop, myop_tmp;
    myop=op1; myop_tmp=op2;
    move_sysop(myop, opid1, opid2, true, stream);
    move_sysop(myop_tmp, opid2, opid2, true, stream);
    myop=myop.mul(myop_tmp);
    
    myop_tmp=op3;
    move_sysop(myop, opid2, opid3, false, stream);
    move_sysop(myop_tmp, opid3, opid3, true, stream);
    myop=myop.mul(myop_tmp);
    
    myop_tmp=op4;
    move_sysop(myop, opid3, opid4, false, stream);
    move_sysop(myop_tmp, opid4, opid4, true, stream);
    myop=myop.mul(myop_tmp);
    move_sysop(myop, opid4, stopid, false, stream);
    
    return myop;
}

reducematrix two_env_site(const reducematrix &op1, const reducematrix &op2, const int &opid1, const int &opid2, cudaStream_t stream[2]) {
	int opid=max(opid1,opid2);
	reducematrix myop1, myop2, myop;
	myop1=op1; myop2=op2;
	move_envop(myop1, opid1, opid, true, stream);
	move_envop(myop2, opid2, opid, true, stream);
	myop=myop1.mul(myop2);
	move_envop(myop, opid, ltot-stopid, false, stream);
    return myop;
}

reducematrix three_env_site(const reducematrix &op1, const reducematrix &op2, const reducematrix &op3, const int &opid1, const int &opid2, const int &opid3, cudaStream_t stream[2]) {
    reducematrix myop, myop_tmp;
    myop=op1; myop_tmp=op2;
    move_envop(myop, opid1, opid2, true, stream);
    move_envop(myop_tmp, opid2, opid2, true, stream);
    myop=myop.mul(myop_tmp);
    
    myop_tmp=op3;
    move_envop(myop, opid2, opid3, false, stream);
    move_envop(myop_tmp, opid3, opid3, true, stream);
    myop=myop.mul(myop_tmp);
    
    move_envop(myop, opid3, ltot-stopid, false, stream);
    return myop;
}

reducematrix four_env_site(const reducematrix &op1, const reducematrix &op2, const reducematrix &op3, const reducematrix &op4, const int &opid1, const int &opid2, const int &opid3, const int &opid4, cudaStream_t stream[2]) {
    reducematrix myop, myop_tmp;
    myop=op1; myop_tmp=op2;
    move_envop(myop, opid1, opid2, true, stream);
    move_envop(myop_tmp, opid2, opid2, true, stream);
    myop=myop.mul(myop_tmp);
    
    myop_tmp=op3;
    move_envop(myop, opid2, opid3, false, stream);
    move_envop(myop_tmp, opid3, opid3, true, stream);
    myop=myop.mul(myop_tmp);
    
    myop_tmp=op4;
    move_envop(myop, opid3, opid4, false, stream);
    move_envop(myop_tmp, opid4, opid4, true, stream);
    myop=myop.mul(myop_tmp);
    move_envop(myop, opid4, ltot-stopid, false, stream);
    
    return myop;
}

double sys_env_site(const reducematrix &op1, const reducematrix &op2, const int &opid1, const int &opid2, cudaStream_t stream[2]) {
	reducematrix myop1, myop2;
	myop1=op1; myop2=op2;
	move_sysop(myop1, opid1, stopid, true, stream);
	move_envop(myop2, opid2, ltot-stopid, true, stream);
    return sys_env_measure(myop1, myop2);
}

double twosite(const reducematrix &op1, const reducematrix &op2, const int &opid1, const int &opid2, cudaStream_t stream[2]) {
    reducematrix myop;
	if (opid1<=stopid) {
		if (opid2<=stopid) {
            myop=two_sys_site(op1, op2, opid1, opid2, stream);
            return sys_measure(myop);
		} else {
			return sys_env_site(op1, op2, opid1, ltot-opid2+1, stream);
		}
	} else {
		if (opid2<=stopid) {
			return sys_env_site(op2, op1, opid2, ltot-opid1+1, stream);
		} else {
            myop=two_env_site(op1, op2, ltot-opid1+1, ltot-opid2+1, stream);
            return env_measure(myop);
		}
	}
}

double sc_helper_4_0(const vector<reducematrix> &myop, const vector<int> &myopid, cudaStream_t stream[2]) {
    double val=0.0;
    reducematrix tmp;
    if (myopid[0]>myopid[2]) {
        if (myopid[0]==myopid[3]) {
            tmp=four_sys_site(myop[2], myop[0], myop[3], myop[1], myopid[2], myopid[0], myopid[3], myopid[1], stream);
            val=-sys_measure(tmp);
        }
        if (myopid[1]==myopid[3]) {
            tmp=four_sys_site(myop[2], myop[0], myop[1], myop[3], myopid[2], myopid[0], myopid[1], myopid[3], stream);
            val=sys_measure(tmp);
        }
    } else {
        if (myopid[1]<=myopid[2]) {
            tmp=four_sys_site(myop[0], myop[1], myop[2], myop[3], myopid[0], myopid[1], myopid[2], myopid[3], stream);
            val=sys_measure(tmp);
        } else if (myopid[1]>myopid[3]) {
            tmp=four_sys_site(myop[0], myop[2], myop[3], myop[1], myopid[0], myopid[2], myopid[3], myopid[1], stream);
            val=sys_measure(tmp);
        } else if (myopid[1]<=myopid[3]) {
            tmp=four_sys_site(myop[0], myop[2], myop[1], myop[3], myopid[0], myopid[2], myopid[1], myopid[3], stream);
            val=-sys_measure(tmp);
        }
    }
    return val;
}

double sc_helper(const reducematrix &op1, const reducematrix &op2, const reducematrix &op3, const reducematrix &op4, const int &opid1, const int &opid2, const int &opid3, const int &opid4, cudaStream_t stream[2]) {
    vector<reducematrix> myop(4);
    vector<int> myopid(4), tmpid(4);
    double mycoef=1.0;
    myop[0]=op1;myop[1]=op2;myop[2]=op3;myop[3]=op4;
    myopid[0]=opid1;myopid[1]=opid2;myopid[2]=opid3;myopid[3]=opid4;
    tmpid=myopid;
    
    sort(tmpid.begin(),tmpid.end());
    if (tmpid[1]>stopid) {
        for (int i=0; i<4; ++i) {
            myopid[i]=ltot-myopid[i]+1;
        }
    }
    
    if (myopid[0]>myopid[1]) {
        reducematrix tmp;
        tmp=myop[0];
        myop[0]=myop[1];
        myop[1]=tmp;
        iter_swap(myopid.begin(), myopid.begin()+1);
        mycoef*=myop[0].sign();
    }
    if (myopid[2]>myopid[3]) {
        reducematrix tmp;
        tmp=myop[2];
        myop[2]=myop[3];
        myop[3]=tmp;
        iter_swap(myopid.begin()+2, myopid.begin()+3);
        mycoef*=myop[2].sign();
    }
    if (myopid[1]!=myopid[3] && myopid[0]!=myopid[3]) {
        if (myopid[0]>myopid[2]) {
            reducematrix tmp;
            tmp=myop[0];
            myop[0]=myop[2];
            myop[2]=tmp;
            iter_swap(myopid.begin(), myopid.begin()+2);
            tmp=myop[1];
            myop[1]=myop[3];
            myop[3]=tmp;
            iter_swap(myopid.begin()+1, myopid.begin()+3);
        }
    }
    
    //-------------------------------------------------------------
    if (tmpid[3]<=stopid) {
        return mycoef*sc_helper_4_0(myop, myopid, stream);
    }
}

void measuresccor(const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    double val;

    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        for (int n = 0; n < fourpoint.size(); ++n) {
            out << "ref=" << fourpoint[n][0].l1 << ", " << fourpoint[n][0].l2 << endl;
            for (int i = 1; i < fourpoint[n].size(); ++i) {
                val=0.0;
                val+= sc_helper(fc_u_d, fc_d_d, fc_d, fc_u, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                val+=-sc_helper(fc_d_d, fc_u_d, fc_d, fc_u, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                val+=-sc_helper(fc_u_d, fc_d_d, fc_u, fc_d, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                val+= sc_helper(fc_d_d, fc_u_d, fc_u, fc_d, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                out << i << " " << n << " " << val << endl;
            }
        }
        out << endl;
        out.close();
    }
}


void measuresccor_quick(const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    
    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        for (int n = 0; n < fourpoint.size(); ++n) {
            out << "ref=" << fourpoint[n][0].l1 << ", " << fourpoint[n][0].l2 << "; " << fourpoint[n][1].l1 << ", " << fourpoint[n][1].l2 << endl;

            if (fourpoint[n][0].l1 < fourpoint[n][0].l2) {
                reducematrix myop, myop_tmp;
                myop=fc_u_d; myop_tmp=fc_d_d;
                move_sysop(myop, fourpoint[n][0].l1, fourpoint[n][0].l2, true, stream);
                move_sysop(myop_tmp, fourpoint[n][0].l2, fourpoint[n][0].l2, true, stream);
                myop=myop.mul(myop_tmp);
                int myloc=fourpoint[n][0].l2;

                double val[fourpoint[n].size()];
                for (int i = 1; i < fourpoint[n].size(); ++i) {
                    if (fourpoint[n][0].l2 < fourpoint[n][i].l1 && fourpoint[n][i].l1<fourpoint[n][i].l2 ) {
                        move_sysop(myop, myloc, fourpoint[n][i].l1, false, stream);
                        myloc=fourpoint[n][i].l1;

                        reducematrix myop1;
                        myop_tmp=fc_d;
                        move_sysop(myop_tmp, fourpoint[n][i].l1, fourpoint[n][i].l1, true, stream);

                        myop1=myop.mul(myop_tmp);
                        move_sysop(myop1, fourpoint[n][i].l1, fourpoint[n][i].l2, false, stream);

                        myop_tmp=fc_u;
                        move_sysop(myop_tmp, fourpoint[n][i].l2, fourpoint[n][i].l2, true, stream);

                        myop1=myop1.mul(myop_tmp);

                        reducematrix myop2;
                        myop_tmp=fc_u;
                        move_sysop(myop_tmp, fourpoint[n][i].l1, fourpoint[n][i].l1, true, stream);

                        myop2=myop.mul(myop_tmp);
                        move_sysop(myop2, fourpoint[n][i].l1, fourpoint[n][i].l2, false, stream);

                        myop_tmp=fc_d;
                        move_sysop(myop_tmp, fourpoint[n][i].l2, fourpoint[n][i].l2, true, stream);

                        myop2=myop2.mul(myop_tmp);

                        myop1.mul_add(-1.0, myop2,stream[0]);

                        move_sysop(myop1, fourpoint[n][i].l2, stopid, false, stream);

                        val[i-1]=sys_measure(myop1);
                        move_sysop(myop, myloc, fourpoint[n][i].l2, false, stream);
                        myloc=fourpoint[n][i].l2;
                    }
                }

                myop=fc_d_d; myop_tmp=fc_u_d;
                move_sysop(myop, fourpoint[n][0].l1, fourpoint[n][0].l2, true, stream);
                move_sysop(myop_tmp, fourpoint[n][0].l2, fourpoint[n][0].l2, true, stream);
                myop=myop.mul(myop_tmp);
                myloc=fourpoint[n][0].l2;

                for (int i = 1; i < fourpoint[n].size(); ++i) {
                    if (fourpoint[n][0].l2 < fourpoint[n][i].l1 && fourpoint[n][i].l1<fourpoint[n][i].l2 ) {
                        move_sysop(myop, myloc, fourpoint[n][i].l1, false, stream);
                        myloc=fourpoint[n][i].l1;

                        reducematrix myop1;
                        myop_tmp=fc_u;
                        move_sysop(myop_tmp, fourpoint[n][i].l1, fourpoint[n][i].l1, true, stream);

                        myop1=myop.mul(myop_tmp);
                        move_sysop(myop1, fourpoint[n][i].l1, fourpoint[n][i].l2, false, stream);

                        myop_tmp=fc_d;
                        move_sysop(myop_tmp, fourpoint[n][i].l2, fourpoint[n][i].l2, true, stream);

                        myop1=myop1.mul(myop_tmp);

                        reducematrix myop2;
                        myop_tmp=fc_d;
                        move_sysop(myop_tmp, fourpoint[n][i].l1, fourpoint[n][i].l1, true, stream);

                        myop2=myop.mul(myop_tmp);
                        move_sysop(myop2, fourpoint[n][i].l1, fourpoint[n][i].l2, false, stream);

                        myop_tmp=fc_u;
                        move_sysop(myop_tmp, fourpoint[n][i].l2, fourpoint[n][i].l2, true, stream);

                        myop2=myop2.mul(myop_tmp);

                        myop1.mul_add(-1.0, myop2,stream[0]);

                        move_sysop(myop1, fourpoint[n][i].l2, stopid, false, stream);

                        val[i-1]+=sys_measure(myop1);
                        move_sysop(myop, myloc, fourpoint[n][i].l2, false, stream);
                        myloc=fourpoint[n][i].l2;

                        // val+= sc_helper(fc_u_d, fc_d_d, fc_d, fc_u, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                        // val+=-sc_helper(fc_u_d, fc_d_d, fc_u, fc_d, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                        // val+=-sc_helper(fc_d_d, fc_u_d, fc_d, fc_u, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                        // val+= sc_helper(fc_d_d, fc_u_d, fc_u, fc_d, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                        out << i << " " << n << " " << val[i-1] << endl;
                    } else {
                        out << i << " " << n << " " << 0.0 << endl;
                    }
                }
            }
        }
        out << endl;
        out.close();
    }
}

void measuresccor_spinless(const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    double val;

    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        for (int n = 0; n < fourpoint.size(); ++n) {
            out << "ref=" << fourpoint[n][0].l1 << ", " << fourpoint[n][0].l2 << endl;
            for (int i = 1; i < fourpoint[n].size(); ++i) {
                val=0.0;
                val+= sc_helper(fc_u_d, fc_u_d, fc_u, fc_u, fourpoint[n][0].l1, fourpoint[n][0].l2, fourpoint[n][i].l1, fourpoint[n][i].l2, stream);
                out << i << " " << n << " " << val << endl;
            }
        }
        out << endl;
        out.close();
    }
}


void measuresccor_spinless_quick(const string &name, cudaStream_t stream[2]) {
    ofstream out(name.c_str(), ios::out | ios::app);
    
    if (out.is_open()) {
        out << scientific;
        out << "ltot = " << ltot << endl;
        for (int n = 0; n < fourpoint.size(); ++n) {
            out << "ref=" << fourpoint[n][0].l1 << ", " << fourpoint[n][0].l2 << endl;

            if (fourpoint[n][0].l1 < fourpoint[n][0].l2) {
                reducematrix myop, myop_tmp;
                myop=fc_u_d; myop_tmp=fc_u_d;
                move_sysop(myop, fourpoint[n][0].l1, fourpoint[n][0].l2, true, stream);
                move_sysop(myop_tmp, fourpoint[n][0].l2, fourpoint[n][0].l2, true, stream);
                myop=myop.mul(myop_tmp);
                int myloc=fourpoint[n][0].l2;

                double val;
                for (int i = 1; i < fourpoint[n].size(); ++i) {
                    if (fourpoint[n][0].l2 < fourpoint[n][i].l1 && fourpoint[n][i].l1<fourpoint[n][i].l2 ) {
                        move_sysop(myop, myloc, fourpoint[n][i].l1, false, stream);
                        myloc=fourpoint[n][i].l1;

                        reducematrix myop1;
                        myop_tmp=fc_u;
                        move_sysop(myop_tmp, fourpoint[n][i].l1, fourpoint[n][i].l1, true, stream);

                        myop1=myop.mul(myop_tmp);
                        move_sysop(myop1, fourpoint[n][i].l1, fourpoint[n][i].l2, false, stream);

                        myop_tmp=fc_u;
                        move_sysop(myop_tmp, fourpoint[n][i].l2, fourpoint[n][i].l2, true, stream);

                        myop1=myop1.mul(myop_tmp);

                        move_sysop(myop1, fourpoint[n][i].l2, stopid, false, stream);

                        val=sys_measure(myop1);
                        out << i << " " << n << " " << val << endl;

                        move_sysop(myop, myloc, fourpoint[n][i].l2, false, stream);
                        myloc=fourpoint[n][i].l2;
                    } else {
                        out << i << " " << n << " " << 0.0 << endl;
                    }
                }
            }
        }
        out << endl;
        out.close();
    }
}