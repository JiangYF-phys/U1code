#include "reducematrix.hpp"
#include "global.hpp"

reducematrix::reducematrix(int size, int sign) {
    mysign=sign;
    mat.reserve(size);
    for (int i=0; i<size; ++i) {
        mat.push_back(new mblock());
    }
}

reducematrix::reducematrix(const reducematrix &rhs):mysign(rhs.mysign) {
    cout << "error reducemat" << endl;
}

reducematrix::reducematrix(const reducematrix &rhs, cudaStream_t stream):mysign(rhs.mysign) {
    for (int i=0; i<rhs.size(); ++i) {
        mat.push_back(new mblock(*rhs.mat[i], stream));
    }
}

reducematrix::~reducematrix() {
    for (size_t i=0; i<mat.size(); ++i) {
        // delete mat[i]; mat[i]=NULL;
        cudaFree(mat[i]->mat);
    }
    mat.clear();
}

void reducematrix::clear() {
    for (size_t i=0; i<mat.size(); ++i) {
        // delete mat[i]; mat[i]=NULL;
        cudaFree(mat[i]->mat);
    }
    mysign=1;
    mat.clear();
}

void reducematrix::setzero(cudaStream_t stream) {
    for (size_t i=0; i<mat.size(); ++i) {
        cudaMemsetAsync(mat[i]->mat, 0, sizeof(double)* mat[i]->sleft*mat[i]->sright, stream);
    }
}

void reducematrix::setran() {
    for (size_t i=0; i<mat.size(); ++i) {
        double *mymat=new double[mat[i]->sleft*mat[i]->sright];
		for (int j = 0; j < mat[i]->sleft*mat[i]->sright; ++j) {
            mymat[j]=(double) rand()/RAND_MAX;
		}
        cudaMemcpy(mat[i]->mat, mymat, mat[i]->sleft*mat[i]->sright*sizeof(double), cudaMemcpyHostToDevice);
        delete [] mymat;
    }
}

void reducematrix::toidentity() {
    for (size_t i=0; i<mat.size(); ++i) {
        double *mymat=new double[mat[i]->sleft*mat[i]->sright];
		for (int j = 0; j < mat[i]->sleft*mat[i]->sright; ++j) {
            mymat[j]=0.0;
		}
        for (int j = 0; j < mat[i]->sright; ++j) {
			mymat[ j*mat[i]->sleft+j ]=1.0;
		}
        cudaMemcpy(mat[i]->mat, mymat, mat[i]->sleft*mat[i]->sright*sizeof(double), cudaMemcpyHostToDevice);
        delete [] mymat;
    }
}

int reducematrix::size() const { return mat.size(); }

int reducematrix::sign() const { return mysign; }
void reducematrix::sign(int sig) { mysign=sig; }

int reducematrix::jmax() const {
    int jm=0;
    for (size_t i=0; i<mat.size(); ++i) {
        if (jm<=max(abs(mat[i]->jleft),abs(mat[i]->jright)) ) {
            jm=max(abs(mat[i]->jleft),abs(mat[i]->jright));
        }
    }
    return jm;
}

int reducematrix::nmax() const {
    int nm=0;
    for (size_t i=0; i<mat.size(); ++i) {
        if ( nm<=max(mat[i]->nleft,mat[i]->nright) ) {
            nm=max(mat[i]->nleft,mat[i]->nright);
        }
    }
    return nm;
}

const int& reducematrix::getjr(int i) const {return mat[i]->jright;}
const int& reducematrix::getnr(int i) const {return mat[i]->nright;}
const int& reducematrix::getsr(int i) const {return mat[i]->sright;}
const int& reducematrix::getjl(int i) const {return mat[i]->jleft;}
const int& reducematrix::getnl(int i) const {return mat[i]->nleft;}
const int& reducematrix::getsl(int i) const {return mat[i]->sleft;}

double reducematrix::mem_size() const {
    double msize=0;
    for (size_t i=0; i<mat.size(); ++i) {
        msize += mat[i]->sleft*mat[i]->sright*sizeof(double);
    }
    return msize/1024/1024;
}

void reducematrix::add(const mblock &block) {
	mblock* tmp=new mblock(block, 0);
    mat.push_back(tmp);
}

void reducematrix::addC(mblock* block) {
    mat.push_back(block);
}

void reducematrix::changemat(const double* val, cudaStream_t stream) {
    int anchor=0;
    for (size_t i=0; i < this->mat.size(); ++i) {
        int mysize = this->mat[i]->sleft * this->mat[i]->sright;
        cudaMemcpyAsync(this->mat[i]->mat, &val[0]+anchor, mysize*sizeof(double), cudaMemcpyHostToDevice, stream);
        anchor+=mysize;
    }    
}

const mblock& reducematrix::get(int i) const {
    assert(i<int(mat.size()));
    return *mat[i];
}

void reducematrix::num_mul_block(int i, double num, cudaStream_t stream) {mat[i]->mul_num(num, stream);}
void reducematrix::num_mul(double num, cudaStream_t stream) {
    for (size_t i = 0; i < mat.size(); i++) {
        mat[i]->mul_num(num, stream);
    }
    
    
}
void reducematrix::mconjtoblock(int i, double num) {mat[i]->mconj(num, 0);}

void reducematrix::mul_add(const double& alpha, const reducematrix& block1, cudaStream_t stream) {
    for (int j=0; j<block1.size(); ++j) {
        bool myflag=true;
        for (int i=0; i<size(); ++i) {
            if (checkplus(*mat[i], *block1.mat[j], 'n') && myflag) {
                dim3 block(32, 32);
                dim3 grid((mat[i]->sleft-1)/block.x+1, (mat[i]->sright-1)/block.y+1);
                matAdd<<<grid,block,0, stream>>>(alpha, mat[i]->mat, mat[i]->sleft, 0, block1.mat[j]->mat, mat[i]->sleft, 0, mat[i]->sleft, mat[i]->sright);
                myflag=false;
            }
        }
        // cudaDeviceSynchronize();

        if (myflag) {
            mblock* tmp=new mblock(*block1.mat[j], stream);
            tmp->mul_num(alpha, stream);
            // *tmp *= alpha;
            mat.push_back(tmp);
        }
    }
}

void reducematrix::addsubblock(int loc, int bgn1, int bgn2,int len1, int len2, const mblock& part) {
    dim3 block(32, 32);
    dim3 grid((len1-1)/block.x+1, (len2-1)/block.y+1);
    matAdd<<<grid,block>>>(1.0, mat[loc]->mat, mat[loc]->sleft, bgn1+bgn2*mat[loc]->sleft, part.mat, len1, 0, len1, len2);
    // cudaDeviceSynchronize();
}

void reducematrix::setblockpart(int i, int partA[4], const mblock& wave, int partB[4], cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((partA[1]-1)/block.x+1, (partA[3]-1)/block.y+1);
    // check
    matcopypart<<<grid,block,0,stream>>>(1.0, this->mat[i]->mat, this->mat[i]->sleft, partA[0], wave.mat, wave.sleft, partB[0], partA[1], partA[3]);
}

void reducematrix::mult_subblock_subblock_rank(int loc, const double alpha, const mblock &block1, const double b2mat, const mblock &wave, const double b3mat, const mblock &block4, double* tmp_mat, const int bgn[4],const char flag[4], cudaStream_t stream) {
    assert(fabs(alpha)>0);
    // env part has an additonal transpose
    int s1, s2, s3, s4;
    cublasOperation_t cb1trans, cb4trans;
    if (flag[0]=='t') {
        s1=block1.sright; s2=block1.sleft;
        cb1trans=CUBLAS_OP_T;
    } else {
        s1=block1.sleft; s2=block1.sright;
        cb1trans=CUBLAS_OP_N;
    }
    if (flag[3]=='t') {
        s3=block4.sleft; s4=block4.sright;
        cb4trans=CUBLAS_OP_N;
    } else {
        s3=block4.sright; s4=block4.sleft;
        cb4trans=CUBLAS_OP_T;
    }

    double zero=0.0, one=1.0;
    double coef;

    if ( strcmp(flag,"niii")==0 ) {
        cublasDgemm(GlobalHandle,cb1trans,CUBLAS_OP_N,s1,s4,s2,&alpha,block1.mat,block1.sleft,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&one,&mat[loc]->mat[0]+bgn[0]+bgn[2]*mat[loc]->sleft,mat[loc]->sleft);
    } else if ( strcmp(flag,"inii")==0 ) {
        if (fabs(b2mat)>0) {
            dim3 block(32, 32);
            dim3 grid((s1-1)/block.x+1, (s4-1)/block.y+1);
            matAdd<<<grid,block,0,stream>>>(alpha*b2mat, mat[loc]->mat, mat[loc]->sleft, bgn[0]+bgn[2]*mat[loc]->sleft, wave.mat, wave.sleft, bgn[1]+bgn[3]*wave.sleft, s1, s4);
            // cudaDeviceSynchronize();
        }
    } else if ( strcmp(flag,"iini")==0) {
        if (fabs(b3mat)>0) {
            dim3 block(32, 32);
            dim3 grid((s1-1)/block.x+1, (s4-1)/block.y+1);
            matAdd<<<grid,block,0,stream>>>(alpha*b3mat, mat[loc]->mat, mat[loc]->sleft, bgn[0]+bgn[2]*mat[loc]->sleft, wave.mat, wave.sleft, bgn[1]+bgn[3]*wave.sleft, s1, s4);
            // cudaDeviceSynchronize();
        }
    } else if ( strcmp(flag,"iiin")==0 ) {
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,cb4trans,s1,s4,s3,&alpha,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&block4.mat[0],block4.sleft,&one,&mat[loc]->mat[0]+bgn[0]+bgn[2]*mat[loc]->sleft,mat[loc]->sleft);
    } else if (flag[0]!='i' && flag[1]!='i') {//ntii tnii
        coef=alpha*b2mat;
        cublasDgemm(GlobalHandle,cb1trans,CUBLAS_OP_N,s1,s4,s2,&coef,&block1.mat[0],block1.sleft,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&one,&mat[loc]->mat[0]+bgn[0]+bgn[2]*mat[loc]->sleft,mat[loc]->sleft);
    } else if (flag[2]!='i' && flag[3]!='i') {//iitn iint
        coef=alpha*b3mat;
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,cb4trans,s1,s4,s3,&coef,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&block4.mat[0],block4.sleft,&one,&mat[loc]->mat[0]+bgn[0]+bgn[2]*mat[loc]->sleft,mat[loc]->sleft);
    } else {
        // double* tmp;
        // cudaMalloc((void**)&tmp, s1*s3*sizeof(double));
        if (flag[0]!='i') {
            assert(flag[1]=='i');
            cublasDgemm(GlobalHandle,cb1trans,CUBLAS_OP_N,s1,s3,s2,&alpha,block1.mat,block1.sleft,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&zero,tmp_mat,s1);
        }
        
        if (flag[1]!='i') {
            assert(flag[0]=='i');
            dim3 block(32, 32);
            dim3 grid((s1-1)/block.x+1, (s3-1)/block.y+1);
            matcopypart<<<grid,block,0,stream>>>(alpha*b2mat, tmp_mat, s1, 0, wave.mat, wave.sleft, bgn[1]+bgn[3]*wave.sleft, s1, s3);
        }
        
        if (flag[2]!='i') {
            assert(flag[3]=='i');
            dim3 block(32, 32);
            dim3 grid((s1-1)/block.x+1, (s3-1)/block.y+1);
            matAdd<<<grid,block,0,stream>>>(b3mat, mat[loc]->mat, mat[loc]->sleft, bgn[0]+bgn[2]*mat[loc]->sleft, tmp_mat, s1, 0, s1, s3);
        }

        if (flag[3]!='i') {
            assert(flag[2]=='i');
            cublasDgemm(GlobalHandle,CUBLAS_OP_N,cb4trans,s1,s4,s3,&one,tmp_mat,s1,&block4.mat[0],block4.sleft,&one,&mat[loc]->mat[0]+bgn[0]+bgn[2]*mat[loc]->sleft,mat[loc]->sleft);
        }
        // cudaFree(tmp);
    }
}

void reducematrix::set(const reducematrix &rhs, cudaStream_t stream) {
    if (this != &rhs) {
        this->clear();
		mysign=rhs.sign();
        this->mat.reserve(rhs.mat.size());
        for (size_t i=0; i<rhs.mat.size(); ++i) {
            mat.push_back(new mblock(*rhs.mat[i], stream));
        }
    }
}

reducematrix& reducematrix::operator =(const reducematrix &rhs) {
    if (this != &rhs) {
        this->clear();
		mysign=rhs.sign();
        this->mat.reserve(rhs.mat.size());
        for (size_t i=0; i<rhs.mat.size(); ++i) {
            mat.push_back(new mblock(*rhs.mat[i], 0));
        }
    }
    return *this;
}

reducematrix& reducematrix::operator +=(const reducematrix &block1) {
    for (int j=0; j<block1.size(); ++j) {
        bool myflag=true;
        for (int i=0; i<size(); ++i) {
            if (checkplus(*mat[i], *block1.mat[j], 'n') && myflag) {
                *mat[i] += *block1.mat[j];
                myflag=false;
            }
        }
        if (myflag) {
            add(block1.get(j));
        }
    }
    return *this;
}

void reducematrix::fromC(const reducematrixCPU &rhs, cudaStream_t stream) {
    this->clear();
    mysign=rhs.sign();
    this->mat.reserve(rhs.size());
    for (size_t i=0; i<rhs.size(); ++i) {
        mblock* tmp = new mblock(*(rhs.mat[i]), stream);
        mat.push_back(tmp);
    }
}

void reducematrix::toCPU(reducematrixCPU &rhs, cudaStream_t stream) const {
    rhs.clear();
    rhs.sign(mysign);
    rhs.reserve(this->size());
    for (size_t i=0; i < this->size(); ++i) {
        mblockCPU* tmp=new mblockCPU();
        (*mat[i]).toCPU(*tmp);
        rhs.mat.push_back(tmp);
    }

    for (size_t i=0; i < this->size(); ++i) {
        cublasGetMatrixAsync(mat[i]->sleft, mat[i]->sright, sizeof(double), mat[i]->mat, mat[i]->sleft, rhs.mat[i]->mat, mat[i]->sleft, stream);
    }

}

int reducematrix::search(const int &jleft, const int &jright, const int &nleft, const int &nright) const {
	vector<mblock*>::const_iterator it = find_if(mat.begin(), mat.end(),[&jleft, &jright, &nleft, &nright] (mblock* const &m) {
		return m->jleft==jleft && m->jright==jright && m->nleft==nleft && m->nright==nright;});
	if (it==mat.end()) {
		return -1;
	} else {
		return distance(mat.begin(), it);
	}
}

void reducematrix::todisk(ofstream& out, const char& flag) const {
    int mysize=mat.size();
    out.write((char*) (&mysize),   sizeof(int));
    out.write((char*) (&mysign),   sizeof(mysign));
    for (int i=0; i<mysize; ++i) {
        mat[i]->todisk(out, flag);
    }
}

void reducematrix::todisk(const string& filename, const char& flag) const {
	ofstream out(filename.c_str(), ios::out | ios::binary | ios::trunc);
	todisk(out, flag);
	out.close();
}

void reducematrix::fromdisk(ifstream& in, const char& flag, cudaStream_t stream) {
    this->clear();
    int mysize;
    in.read((char*) (&mysize),  sizeof(mysize));
	in.read((char*) (&mysign),  sizeof(mysign));
    if (flag=='n' || flag=='u') {
        for (int i=0; i<mysize; ++i) {
            mblock* tmp=new mblock(in, flag, stream);
            mat.push_back(tmp);
        }
    }
}

void reducematrix::fromdisk(const string& filename, const char& flag, cudaStream_t stream) {
	ifstream in(filename.c_str(), ios::in | ios::binary);
	fromdisk(in, flag, stream);
	in.close();
}

reducematrix reducematrix::wavemul(const reducematrix &block1, char flag1, char flag2, cudaStream_t stream) const {
    reducematrix mulm(0, 1);
    for (size_t i=0; i<mat.size(); ++i) {
        if ( checktime(*mat[i],*block1.mat[i],flag1,flag2) ) {
            
            int loc;
            if (flag1=='n') {
                loc=mulm.search(mat[i]->jleft, block1.mat[i]->jleft, mat[i]->nleft, block1.mat[i]->nleft);
            } else {
                loc=mulm.search(mat[i]->jright, block1.mat[i]->jright, mat[i]->nright, block1.mat[i]->nright);
            }
            if (loc==-1) {
                mulm.mat.push_back(new mblock());
                mulm.mat.back()->mult(1,*mat[i], *block1.mat[i], flag1, flag2);
            } else {
                // *mulm.mat[loc] += *mat[i] * *block1.mat[i];
                mulm.mat[loc]->addto(multwithrank(1, *mat[i], *block1.mat[i], flag1, flag2), stream);
            }
            
        }
    }
    return mulm;
}

reducematrix reducematrix::applytrunc(const reducematrix &trunc, cudaStream_t stream) {
    reducematrix mulm(0,mysign);
    for (size_t i=0; i<mat.size(); ++i) {
        for (int l=0; l<trunc.size(); ++l) {
            for (int r=0; r<trunc.size(); ++r) {
                if (checktime(*mat[i], *trunc.mat[r],'n','n') && checktime(*trunc.mat[l], *mat[i],'t','n')) {
                    mblock* tmp=new mblock(multwithrank(1, *trunc.mat[l], *mat[i] * *trunc.mat[r],'t','n'), stream);
                    mulm.mat.push_back(tmp);
                    // mulm.add(multwithrank(1, *trunc.mat[l], *mat[i] * *trunc.mat[r],'t','n'));
                }
            }
        }
    }
    return mulm;
}

void reducematrix::trunc(const reducematrix &trunc) {
    int count=0;
    for (size_t i=0; i<mat.size(); ++i) {
        int ct_in=0;
        for (int l=0; l<trunc.size(); ++l) {
            for (int r=0; r<trunc.size(); ++r) {
                if (checktime(*mat[i], *trunc.mat[r],'n','n') && checktime(*trunc.mat[l], *mat[i],'t','n')) {
                    // *mat[count]=multwithrank(1, *trunc.mat[l], *mat[i] * *trunc.mat[r],'t','n');
                    mat[count]->mult(1, *trunc.mat[l], *mat[i] * *trunc.mat[r],'t','n');
                    ct_in++;
                    count++;
                }
            }
        }
        if (ct_in>1) {cout << "error in applytrunc" << endl;}
    }
    int msize=mat.size();
    for (size_t i=count; i<msize; ++i) {
        cudaFree(mat[i]->mat); 
    }
    for (size_t i=count; i<msize; ++i) {
        mat.pop_back();
    }
}

vector<repmap> jmap(const reducematrix &ham1, const reducematrix &ham2) {
    vector<repmap> map; map.clear();
    repmap tmp;
    int jmax=ham1.jmax()+ham2.jmax();
    int nmax=ham1.nmax()+ham2.nmax();
    int count, num=0;
    for ( int myj=-jmax; myj<=jmax; ++myj) {
    for ( int myn=0; myn<=nmax; ++myn) {
        count=0;
        for (int i=0; i<ham1.size(); ++i) {
            for (int j=0; j<ham2.size(); ++j) {
                tmp.j1=ham1.mat[i]->jleft;
                tmp.j2=ham2.mat[j]->jleft;
                tmp.n1=ham1.mat[i]->nleft;
                tmp.n2=ham2.mat[j]->nleft;
                if ( myj == tmp.j1 + tmp.j2 && tmp.n1+tmp.n2==myn ) {
                    tmp.j=myj; tmp.n=myn;
                    tmp.len=ham1.mat[i]->sleft*ham2.mat[j]->sleft;
                    map.push_back(tmp);
                    count++;
                }
            }
        }
        if (count>0) {
            int loc=0;
            for (int i=count; i>0; i--) {
                map[map.size()-i].bgn=loc;
                loc += map[map.size()-i].len;
            }
            for (int i=count; i>0; i--) {
                map[map.size()-i].end=loc;
            }
            num++;
        }
    }
    }
    sort(map.begin(),map.end());
    return map;
}

reducematrix operator *(const double &num, const reducematrix &block1) {
    reducematrix tim(block1, 0);
    for (int i=0; i<tim.size(); ++i) {
        tim.mat[i]->mul_num(num, 0) ;
    }
    return tim;
}

bool prodhelp1(const reducematrix &block, int j1, int j2, int n1, int n2, int &loc) {
    for (int i=0; i<block.size(); ++i) {
        if (block.getjl(i)==j1 && block.getjr(i)==j2 && block.getnl(i)==n1 && block.getnr(i)==n2) {
            loc=i;
            return false;
        }
    }
    return true;
}

void reducematrix::prod(const reducematrix &block1, const reducematrix &block2, const vector<repmap> &map, double para, const char &flag1, const char &flag2) {
    if (mat.size()==0) {
        mysign=block1.sign()*block2.sign();
    }
    assert(block1.sign()*block2.sign()==mysign);
    for (int i=0; i<block1.size(); ++i) {
        int jl1, jr1, nl1, nr1;
        if (flag1=='t') {
            jl1=block1.mat[i]->jright; nl1=block1.mat[i]->nright;jr1=block1.mat[i]->jleft; nr1=block1.mat[i]->nleft;
        } else {
            jl1=block1.mat[i]->jleft; nl1=block1.mat[i]->nleft;jr1=block1.mat[i]->jright; nr1=block1.mat[i]->nright;
        }
        for (int j=0; j<block2.size(); ++j) {
            int jl2, jr2, nl2, nr2;
            if (flag2=='t') {
                jl2=block2.mat[j]->jright;nl2=block2.mat[j]->nright;jr2=block2.mat[j]->jleft;nr2=block2.mat[j]->nleft;
            } else {
                jl2=block2.mat[j]->jleft;nl2=block2.mat[j]->nleft;jr2=block2.mat[j]->jright;nr2=block2.mat[j]->nright;
            }
            int myn2=nr1 + nr2; int myn1=nl1 + nl2;
            int myj1 = jl1 + jl2; int myj2 = jr1 + jr2;
            int i1=searchmap(map, jl1, jl2, myj1, nl1, nl2, myn1);
            if (i1>=0) {
                int i2=searchmap(map, jr1, jr2, myj2, nr1, nr2, myn2);
                if (i2>=0) {
                    double j9=1;
                    int loc;
                    if (prodhelp1(*this, myj1, myj2, myn1, myn2, loc)) {
                        mblock* tmp = new mblock( myj1, myj2, map[i1].end, map[i2].end, myn1, myn2);
                        cudaMemset(tmp->mat, 0, sizeof(double)*map[i1].end*map[i2].end);
                        mat.push_back(tmp);
                        loc=size()-1;
                    }
                    j9*=para * pow(block2.sign(),nr1 % 2);
                    mat[loc]->AddProd(j9, *block1.mat[i], *block2.mat[j],map[i1].bgn, map[i2].bgn, flag1, flag2);
                }
            }
        }
    }
}

void reducematrix::prod_id(const reducematrix &block1, const reducematrix &block2, const vector<repmap> &map, const double &para, const char &id_pos, cudaStream_t stream) {
    if (mat.size()==0) {
        mysign=block1.sign()*block2.sign();
    }
    assert(block1.sign()*block2.sign()==mysign);
    for (int i=0; i<block1.size(); ++i) {
        for (int j=0; j<block2.size(); ++j) {
            int myn1=block1.mat[i]->nleft + block2.mat[j]->nleft;
            int myn2=block1.mat[i]->nright + block2.mat[j]->nright;
            int myj1=block1.mat[i]->jleft + block2.mat[j]->jleft;
            int i1=searchmap(map, block1.mat[i]->jleft , block2.mat[j]->jleft , myj1, block1.mat[i]->nleft , block2.mat[j]->nleft , myn1);
            if (i1>=0) {
                int myj2 = block1.mat[i]->jright + block2.mat[j]->jright;
                int i2=searchmap(map, block1.mat[i]->jright, block2.mat[j]->jright, myj2, block1.mat[i]->nright, block2.mat[j]->nright, myn2);
                if (i2>=0) {
                    int loc;
                    double j9=1;
                    if (prodhelp1(*this, myj1, myj2, myn1, myn2, loc)) {
                        mblock* tmp=new mblock( myj1, myj2, map[i1].end, map[i2].end, myn1, myn2);
                        cudaMemsetAsync(tmp->mat, 0, sizeof(double)*map[i1].end*map[i2].end, stream);
                        mat.push_back(tmp);
                        loc=size()-1;
                    }
                    j9*=para * pow(block2.sign(),block1.mat[i]->nright % 2);
                    if (id_pos=='r') {
                        mat[loc]->Add_Prod_Id(j9, *block1.mat[i], block2.mat[j]->sleft, map[i1].bgn, map[i2].bgn, id_pos, stream);
                    } else if (id_pos=='l') {
                        mat[loc]->Add_Prod_Id(j9, *block2.mat[j], block1.mat[i]->sleft, map[i1].bgn, map[i2].bgn, id_pos, stream);
                    }
                }
            }
        }
    }
}

reducematrix reducematrix::conj(cudaStream_t stream) const{
    reducematrix cj(*this, stream);
    for (size_t i=0; i < mat.size(); ++i) {
        double coef=1;
        cj.mat[i]->mconj(coef, stream);
    }
    return cj;
}

ostream& operator<<(ostream& out, const reducematrix& block) {
    out << endl;
    cout << "size=" << block.size() << ", sign=" << block.sign()<< endl;
    for (int i=0; i<block.size(); ++i) {
        cout << block.get(i);
    }
    return out;
}


//========================================================================================================================
//                                                      lagacy
//========================================================================================================================
// only appear in measurement

mblock kmult (mblock* const &block1, mblock* const &block2) {
    mblock block0;
    block0=*block1 * *block2;
    return block0;
}

reducematrix reducematrix::mul(const reducematrix &block1) const {
    reducematrix mulm(0,mysign*block1.sign());
    for (size_t i=0; i<mat.size(); ++i) {
        for (int j=0; j<block1.size(); ++j) {
            if (checktime(*mat[i], * block1.mat[j],'n','n')) {
                int loc=mulm.search(mat[i]->jleft, block1.mat[j]->jright, mat[i]->nleft, block1.mat[j]->nright);
                if (loc==-1) {
                    mulm.add( kmult(mat[i],block1.mat[j]) );
                } else {
                    *mulm.mat[loc] += kmult(mat[i],block1.mat[j]);
                }
            }
        }
    }
    return mulm;
}

void reducematrix::mult_block_rank(int loc, double alpha, const mblock &block1, const mblock &block2, char flag1, char flag2) {
    int s1,s2;
    if (flag1=='n') {
        s1=block1.sleft;
    } else {
        s1=block1.sright;
    }
    if (flag2=='n') {
        s2=block2.sright;
    } else {
        s2=block2.sleft;
    }
    
    double one=1.0;
    if (checktime(block1,block2,flag1,flag2)) {
        if (flag1=='n' && flag2=='n') {
            cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_N,s1,s2,block1.sright,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&one,mat[loc]->mat,s1);
        } else if (flag1=='n' && flag2=='t') {
            cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_T,s1,s2,block1.sright,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&one,mat[loc]->mat,s1);
        } else if (flag1=='t' && flag2=='n') {
            cublasDgemm(GlobalHandle,CUBLAS_OP_T,CUBLAS_OP_N,s1,s2,block1.sleft,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&one,mat[loc]->mat,s1);
        } else {
            cublasDgemm(GlobalHandle,CUBLAS_OP_T,CUBLAS_OP_T,s1,s2,block1.sleft,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&one,mat[loc]->mat,s1);
        }
    }
}

void reducematrix::mult_block_block_rank(int loc, double alpha, const mblock &block1, const mblock &wave, const mblock &block2, char flag1, char flag2) {
    assert(flag1 == 'n'); assert(flag2 == 't');
    int s1,s2;
    if (flag1=='n') {
        s1=block1.sleft;
    } else {
        s1=block1.sright;
    }
    if (flag2=='n') {
        s2=block2.sright;
    } else {
        s2=block2.sleft;
    }
    
    if (checktime(block1,wave,flag1,'n') && checktime(wave,block2,'n',flag2) ) {
        double one=1.0, zero=0.0;
        double* tmp;
        cudaMalloc((void**)&tmp, s1*wave.sright*sizeof(double));
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_N,s1,wave.sright,block1.sright,&one,block1.mat,block1.sleft,wave.mat,wave.sleft,&zero,tmp,s1);
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_T,s1,s2,wave.sright,&alpha,tmp,s1,block2.mat,block2.sleft,&one,mat[loc]->mat,s1);
        cudaFree(tmp);
        // cublasDestroy(handle);
    }
}

//========================================================================================================================

reducematrixCPU::reducematrixCPU(int size, int sign) {
    mysign=sign;
    mat.reserve(size);
    for (int i=0; i<size; ++i) {
        mat.push_back(new mblockCPU());
    }
}

reducematrixCPU::~reducematrixCPU() {
    for (size_t i=0; i<mat.size(); ++i) {
        delete mat[i]; 
        // cudaFreeHost(mat[i]->mat);
    }
    mat.clear();
}

int reducematrixCPU::size() const { return mat.size(); }
int reducematrixCPU::sign() const { return mysign; }
void reducematrixCPU::sign(int sig) { mysign=sig; }
void reducematrixCPU::reserve(int size) {mat.reserve(size);}

void reducematrixCPU::clear() {
    for (size_t i=0; i<mat.size(); ++i) {
        delete mat[i];
        // cudaFreeHost(mat[i]->mat);
        // mat[i]->mat=NULL;
    }
    mysign=1;
    mat.clear();
}

void reducematrixCPU::add(const mblockCPU &block) {
	mblockCPU* tmp=new mblockCPU(block);
    mat.push_back(tmp);
}

const mblockCPU& reducematrixCPU::get(int i) const {
    assert(i<int(mat.size()));
    return *mat[i];
}

void reducematrixCPU::fromdisk(ifstream& in, const char& flag) {
    int mysize;
    in.read((char*) (&mysize),  sizeof(mysize));
	in.read((char*) (&mysign),  sizeof(mysign));
    mat.clear();
    if (flag=='n' || flag=='u') {
        for (int i=0; i<mysize; ++i) {
            mblockCPU* tmp=new mblockCPU(in, flag);
            mat.push_back(tmp);
        }
    }

}

void reducematrixCPU::fromdisk(const string& filename, const char& flag) {
	ifstream in(filename.c_str(), ios::in | ios::binary);
	fromdisk(in, flag);
	in.close();
}

double reducematrixCPU::mem_size() const {
    double msize=0;
    for (size_t i=0; i<mat.size(); ++i) {
        msize += mat[i]->sleft*mat[i]->sright*sizeof(double);
    }
    return msize/1024/1024;
}

ostream& operator<<(ostream& out, const reducematrixCPU& block) {
    out << endl;
    cout << "size=" << block.size() << ", sign=" << block.sign()<< endl;
    for (int i=0; i<block.size(); ++i) {
        cout << block.get(i);
    }
    return out;
}