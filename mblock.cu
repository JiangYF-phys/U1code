#include "mblock.hpp"
#include "global.hpp"

__global__ void matcopy(double* A, double* B, int s1) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<s1) {
        B[i] = A[i];
    }
}

mblock::mblock(const mblock& block, cudaStream_t stream):jleft(block.jleft),nleft(block.nleft),sleft(block.sleft),jright(block.jright),nright(block.nright),sright(block.sright) {
    cudaMalloc((void**)&mat, sleft*sright*sizeof(double));
    int blockSize = 256;
    matcopy<<<(sleft*sright-1)/blockSize+1, blockSize, 0, stream>>>(block.mat, mat, sleft*sright);
    // cudaDeviceSynchronize();
}

mblock::mblock(const mblock& block):jleft(block.jleft),nleft(block.nleft),sleft(block.sleft),jright(block.jright),nright(block.nright),sright(block.sright) {
    cout << "error mblock" << endl;
}

mblock::mblock(const mblockCPU& block, cudaStream_t stream):jleft(block.jleft),nleft(block.nleft),sleft(block.sleft),jright(block.jright),nright(block.nright),sright(block.sright) {
    cudaMalloc((void**)&mat, sleft*sright*sizeof(double));
    cublasSetMatrixAsync(sleft, sright, sizeof(*(block.mat)), block.mat, sleft, mat, sleft, stream);
}

mblock::mblock(const int& jleft, const int& jright, const int& sleft, const int& sright, const int& nleft, const int& nright) {
    this->jleft  = jleft; this->jright = jright;
    this->sleft  = sleft; this->sright = sright;
    this->nleft  = nleft; this->nright = nright;
	// mat = new double[sleft*sright];
    cudaMalloc((void**)&mat, sleft*sright*sizeof(double));
}

mblock::mblock(ifstream& in, const char& flag, cudaStream_t stream) {
    in.read((char*) (&jleft),  sizeof(jleft));
    in.read((char*) (&jright), sizeof(jright));
    in.read((char*) (&nleft),  sizeof(nleft));
    in.read((char*) (&nright), sizeof(nright));
    in.read((char*) (&sleft),  sizeof(sleft));
    in.read((char*) (&sright), sizeof(sright));
    cudaMalloc((void**)&mat, sleft*sright*sizeof(double));

    double *mymat = new double[sleft*sright];
    if (flag=='n' || (flag=='s' && jleft < jright)) {
        in.read((char*) &mymat[0], sleft*sright*sizeof(mymat[0]));
    } else if (flag=='u') {
        for (int i=0; i<sright; ++i) {
            in.read((char*) &mymat[(1+sleft)*i], (sleft-i)*sizeof(mymat[0]));
            for (int j=i+1; j<sleft; ++j) {
                mymat[sleft*j+i]=mymat[sleft*i+j];
            }
        }
    }

    cublasSetMatrixAsync(sleft, sright, sizeof(*mymat), mymat, sleft, mat, sleft, stream);
    delete [] mymat;
}

mblock::~mblock() { 
    //delete [] mat; mat=NULL;
    cudaFree(mat); 
}

void mblock::todisk(ofstream& out, const char& flag) const {
    if (flag=='n' || flag=='u' || (flag=='s' && jleft <= jright)) {
        out.write((char*) (&jleft),  sizeof(jleft));
        out.write((char*) (&jright), sizeof(jright));
        out.write((char*) (&nleft),  sizeof(nleft));
        out.write((char*) (&nright), sizeof(nright));
        out.write((char*) (&sleft),  sizeof(sleft));
        out.write((char*) (&sright), sizeof(sright));
        
        double *mymat = new double[sleft*sright];
        cublasGetMatrix(sleft, sright, sizeof(*mymat), mat, sleft, mymat, sleft);
        
        if (flag=='n' || (flag=='s' && jleft < jright) ) {
            out.write((char*) &mymat[0], sright*sleft*sizeof(mymat[0]));
        } else if (flag=='u' || (flag=='s' && jleft == jright) ) {
            for (int i=0; i<sright; ++i) {
                out.write((char*) &mymat[(1+sleft)*i], (sleft-i)*sizeof(mymat[0]));
            }
        }
        delete [] mymat;
    }
}

void mblock::set(const mblock& block) {
    this->jleft  = block.jleft; this->jright = block.jright;
    this->nleft  = block.nleft; this->nright = block.nright;
}

void mblock::set(int jleft, int jright, int nleft, int nright) {
    this->jleft  = jleft; this->jright = jright;
    this->nleft  = nleft; this->nright = nright;
}

mblock& mblock::operator =(const mblock &rhs) {
    if (this != &rhs) {
        jleft  = rhs.jleft; jright = rhs.jright;
        sleft  = rhs.sleft; sright = rhs.sright;
        nleft  = rhs.nleft; nright = rhs.nright;
        cudaFree(mat);
        cudaMalloc((void**)&mat, sleft*sright*sizeof(double));
        int blockSize = 256;
        matcopy<<<(sleft*sright-1)/blockSize+1, blockSize>>>(rhs.mat, mat, sleft*sright);
        // cudaDeviceSynchronize();
    }
    return *this;
}

__global__ void matscal(const double alpha, double* A, int s1) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<s1) {
        A[i] *= alpha;
    }
}

void mblock::mul_num(const double alpha, cudaStream_t stream) {
    int blockSize = 256;
    matscal<<<(sleft*sright-1)/blockSize+1, blockSize, 0, stream>>>( alpha, mat, sleft*sright);
    // cublasDscal(GlobalHandle,sleft*sright,&alpha,mat,1);
}

// mblock& mblock::operator *=(const double alpha) {
//     int blockSize = 256;
//     matscal<<<(sleft*sright-1)/blockSize+1, blockSize>>>( alpha, mat, sleft*sright);
//     return *this;
// }

void mblock::mconj(const double alpha, cudaStream_t stream){// to be improved
    // mkl_dimatcopy('c', 't', sleft, sright, alpha, mat, sleft, sright);
    double beta=0.0;
    double *tmat;
    cudaMalloc((void**)&tmat, sleft*sright*sizeof(double));
    cublasDgeam(GlobalHandle, CUBLAS_OP_T, CUBLAS_OP_N, sright, sleft, &alpha, mat, sleft, &beta, mat, sright, tmat, sright);
    int blockSize = 256;
    matcopy<<<(sleft*sright-1)/blockSize+1, blockSize, 0, stream>>>(tmat, mat, sleft*sright);
    // cudaDeviceSynchronize();
    cudaFree(tmat);
    // time_1+=1;

    int tmp;
	tmp = jright; jright = jleft; jleft = tmp;
    tmp=sright; sright=sleft; sleft=tmp;
    tmp=nright; nright=nleft; nleft=tmp;
}

mblock mconj(const mblock &block1, const double alpha){
    mblock cj(block1.jright, block1.jleft, block1.sright, block1.sleft, block1.nright, block1.nleft);
    // mkl_domatcopy ('c', 't', block1.sleft, block1.sright, alpha, block1.mat, block1.sleft, cj.mat, block1.sright);
    double beta=0.0;
    cublasDgeam(GlobalHandle, CUBLAS_OP_T, CUBLAS_OP_N, block1.sright, block1.sleft, &alpha, block1.mat, block1.sleft, &beta, block1.mat, block1.sright, cj.mat, block1.sright);
    return cj;
}

void mblock::fromC(const mblockCPU &blockC) {
    jleft  = blockC.jleft; jright = blockC.jright;
    sleft  = blockC.sleft; sright = blockC.sright;
    nleft  = blockC.nleft; nright = blockC.nright;
    cudaFree(mat);
    cudaMalloc((void**)&mat, sleft*sright*sizeof(double));
    cublasSetMatrix(sleft, sright, sizeof(double), blockC.mat, sleft, mat, sleft);
}

void mblock::toCPU(mblockCPU &blockC) {
    blockC.jleft = jleft; blockC.jright = jright;
    blockC.sleft = sleft; blockC.sright = sright;
    blockC.nleft = nleft; blockC.nright = nright;
    delete[] blockC.mat;
    // cudaFreeHost(blockC.mat);
    blockC.mat = new double[sleft*sright];
    // cudaMallocHost(&(blockC.mat), sleft*sright*sizeof(double));
    // cublasGetMatrixAsync(sleft, sright, sizeof(double), mat, sleft, blockC.mat, sleft, stream);
}

__global__ void matAdd(double alpha, double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<s1 && j<s2) {
        A[posA+i+j*LDA] += alpha*B[posB+i+j*LDB];
    }
}

__global__ void matcopypart(double alpha, double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<s1 && j<s2) {
        A[posA+i+j*LDA] = alpha*B[posB+i+j*LDB];
    }
}

bool checktime(const mblock &block1, const mblock &block2, char flag1, char flag2) {
    int s1,s2,n1,n2,j1,j2;
    if (flag1=='n') {
        s1=block1.sright; j1=block1.jright ; n1=block1.nright;
    } else {
        s1=block1.sleft; j1=block1.jleft; n1=block1.nleft;
    }
    
    if (flag2=='n') {
        s2=block2.sleft; j2=block2.jleft; n2=block2.nleft;
    } else {
        s2=block2.sright; j2=block2.jright ; n2=block2.nright;
    }

    if (s1 != s2 || j1 != j2 || n1 != n2) {
        return false;
    }
    return true;
}

bool checkplus(const mblock &block1, const mblock &block2, char flag) {
    if (flag=='n') {
        if (block1.nright == block2.nright && block1.nleft == block2.nleft) {
            if (block1.jright == block2.jright && block1.jleft == block2.jleft) {
                assert(block1.sright == block2.sright && block1.sleft == block2.sleft);
                return true;
            }
        }
        return false;
    } else {
        if (block1.nright == block2.nleft && block1.nleft == block2.nright) {
            if (block1.jright == block2.jleft && block1.jleft == block2.jright) {
                assert(block1.sright == block2.sleft && block1.sleft == block2.sright);
                return true;
            }
        }
        return false;
    }
}

mblock mblock::operator *(const mblock &block1) const {
    assert(checktime(*this, block1, 'n', 'n'));
    mblock block0(jleft,block1.jright,sleft,block1.sright,nleft,block1.nright);
    double alpha=1.0, beta=0.0;
    cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_N,sleft,block1.sright,sright,&alpha,mat,sleft,block1.mat,block1.sleft,&beta,block0.mat,sleft);    

    return block0;
}

void mblock::mult(double alpha, const mblock &block1, const mblock &block2, char flag1, char flag2) {
    if (flag1=='n') {
        jleft=block1.jleft; sleft=block1.sleft; nleft=block1.nleft;
    } else {
        jleft=block1.jright; sleft=block1.sright; nleft=block1.nright;
    }
    if (flag2=='n') {
        jright=block2.jright; sright=block2.sright; nright=block2.nright;
    } else {
        jright=block2.jleft; sright=block2.sleft; nright=block2.nleft;
    }
    
    assert(checktime(block1,block2,flag1,flag2));
    // mblock block0(j1,j2,s1,s2,n1,n2);
    cudaFree(mat);
    cudaMalloc((void**)&mat, sleft*sright*sizeof(double));

    double beta=0.0;
    if (flag1=='n' && flag2=='n') {
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_N,sleft,sright,block1.sright,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,mat,sleft);
    } else if (flag1=='n' && flag2=='t') {
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_T,sleft,sright,block1.sright,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,mat,sleft);
    } else if (flag1=='t' && flag2=='n') {
        cublasDgemm(GlobalHandle,CUBLAS_OP_T,CUBLAS_OP_N,sleft,sright,block1.sleft,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,mat,sleft);
    } else {
        cublasDgemm(GlobalHandle,CUBLAS_OP_T,CUBLAS_OP_T,sleft,sright,block1.sleft,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,mat,sleft);
    }

}

mblock multwithrank(double alpha, const mblock &block1, const mblock &block2, char flag1, char flag2) {
    int s1,s2,n1,n2,j1,j2;
    if (flag1=='n') {
        j1=block1.jleft; s1=block1.sleft; n1=block1.nleft;
    } else {
        j1=block1.jright; s1=block1.sright; n1=block1.nright;
    }
    if (flag2=='n') {
        j2=block2.jright; s2=block2.sright; n2=block2.nright;
    } else {
        j2=block2.jleft; s2=block2.sleft; n2=block2.nleft;
    }
    
    assert(checktime(block1,block2,flag1,flag2));
    mblock block0(j1,j2,s1,s2,n1,n2);

    double beta=0.0;
    if (flag1=='n' && flag2=='n') {
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_N,s1,s2,block1.sright,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,block0.mat,s1);
    } else if (flag1=='n' && flag2=='t') {
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,CUBLAS_OP_T,s1,s2,block1.sright,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,block0.mat,s1);
    } else if (flag1=='t' && flag2=='n') {
        cublasDgemm(GlobalHandle,CUBLAS_OP_T,CUBLAS_OP_N,s1,s2,block1.sleft,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,block0.mat,s1);
    } else {
        cublasDgemm(GlobalHandle,CUBLAS_OP_T,CUBLAS_OP_T,s1,s2,block1.sleft,&alpha,block1.mat,block1.sleft,block2.mat,block2.sleft,&beta,block0.mat,s1);
    }

    return block0;
}

void mblock::mult_subblock_subblock_rank(const double alpha, const mblock &block1, const double b2mat, const mblock &wave, const double b3mat, const mblock &block4, double* tmp_mat, const int bgn[4],const char flag[4], cudaStream_t stream) {
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
        cublasDgemm(GlobalHandle,cb1trans,CUBLAS_OP_N,s1,s4,s2,&alpha,block1.mat,block1.sleft,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&one,&mat[0]+bgn[0]+bgn[2]* sleft,sleft);
    } else if ( strcmp(flag,"inii")==0 ) {
        if (fabs(b2mat)>0) {
            dim3 block(32, 32);
            dim3 grid((s1-1)/block.x+1, (s4-1)/block.y+1);
            matAdd<<<grid,block,0,stream>>>(alpha*b2mat, mat, sleft, bgn[0]+bgn[2]* sleft, wave.mat, wave.sleft, bgn[1]+bgn[3]*wave.sleft, s1, s4);
            // cudaDeviceSynchronize();
        }
    } else if ( strcmp(flag,"iini")==0) {
        if (fabs(b3mat)>0) {
            dim3 block(32, 32);
            dim3 grid((s1-1)/block.x+1, (s4-1)/block.y+1);
            matAdd<<<grid,block,0,stream>>>(alpha*b3mat, mat, sleft, bgn[0]+bgn[2]* sleft, wave.mat, wave.sleft, bgn[1]+bgn[3]*wave.sleft, s1, s4);
            // cudaDeviceSynchronize();
        }
    } else if ( strcmp(flag,"iiin")==0 ) {
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,cb4trans,s1,s4,s3,&alpha,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&block4.mat[0],block4.sleft,&one,&mat[0]+bgn[0]+bgn[2]*sleft, sleft);
    } else if (flag[0]!='i' && flag[1]!='i') {//ntii tnii
        coef=alpha*b2mat;
        cublasDgemm(GlobalHandle,cb1trans,CUBLAS_OP_N,s1,s4,s2,&coef,&block1.mat[0],block1.sleft,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&one,&mat[0]+bgn[0]+bgn[2]* sleft, sleft);
    } else if (flag[2]!='i' && flag[3]!='i') {//iitn iint
        coef=alpha*b3mat;
        cublasDgemm(GlobalHandle,CUBLAS_OP_N,cb4trans,s1,s4,s3,&coef,&wave.mat[0]+bgn[1]+bgn[3]*wave.sleft,wave.sleft,&block4.mat[0],block4.sleft,&one,&mat[0]+bgn[0]+bgn[2]* sleft, sleft);
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
            matAdd<<<grid,block,0,stream>>>(b3mat, mat, sleft, bgn[0]+bgn[2]* sleft, tmp_mat, s1, 0, s1, s3);
        }

        if (flag[3]!='i') {
            assert(flag[2]=='i');
            cublasDgemm(GlobalHandle,CUBLAS_OP_N,cb4trans,s1,s4,s3,&one,tmp_mat,s1,&block4.mat[0],block4.sleft,&one,&mat[0]+bgn[0]+bgn[2]* sleft, sleft);
        }
        // cudaFree(tmp);
    }
}


__global__ void matAddpart(double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<s1 && j<s2) {
        B[posB+i+j*LDB] = A[posA+i+j*LDA];
    }
}

mblock mblock::block(int bgn1, int bgn2,int len1, int len2) { // to be improved
	mblock part(jleft,jright,len1,len2,nleft,nright);
    dim3 blockcu(32, 32);
    dim3 grid( (len1-1)/blockcu.x+1, (len2-1)/blockcu.y+1 );
    matAddpart<<<grid,blockcu>>>(mat, sleft, bgn1+bgn2*sleft, part.mat, len1, 0, len1, len2);
    // cudaDeviceSynchronize();
    return part;
}

double mblock::norm() const {
	double nor=0;
    // nor=cblas_ddot(sleft*sright, mat, 1, mat, 1);
    cublasDdot(GlobalHandle,sleft*sright,mat,1,mat,1,&nor);
	return sqrt(nor);
}

void mblock::AddProd(double alpha, const mblock &block1, const mblock &block2, const int bgn1, const int bgn2, char flag1, char flag2) {
	assert(flag2 == 'n');
	assert(flag1 == 'n' || flag1 =='t');
	if (flag1=='n' && flag2=='n') {
        for (int j1 = 0; j1 < block1.sright; ++j1) {
			for (int j2 = 0; j2 < block2.sright; ++j2) {
                cublasDger(GlobalHandle,block2.sleft, block1.sleft,&alpha,&(block2.mat[0]) + j2*block2.sleft, 1, 
				&(block1.mat[0]) + j1*block1.sleft, 1, &mat[0]+(bgn2 + j1*block2.sright + j2)*sleft + bgn1, block2.sleft);
            }
        }
    }
    
    if (flag1=='t' && flag2=='n') {
        for (int i1 = 0; i1 < block1.sleft; ++i1) {
			for (int j2 = 0; j2 < block2.sright; ++j2) {
                cublasDger(GlobalHandle,block2.sleft, block1.sleft,&alpha,&(block2.mat[0]) + j2*block2.sleft, 1, 
				&(block1.mat[0]) + i1, block1.sleft, &mat[0]+(bgn2 + i1*block2.sright+j2)*sleft + bgn1, block2.sleft);
            }
        }
    }
}

__global__ void prodhelp_l(double alpha, double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2, int i_size) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i<s1 && j<s2 && k<i_size) {
        A[posA+k*(s1+s2*LDA)+i+j*LDA] += alpha*B[posB+i+j*LDB];
    }
}

__global__ void prodhelp_r(double alpha, double* A, int LDA, int posA, double* B, int LDB, int posB, int s1, int s2, int i_size) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i<s1 && j<s2 && k<i_size) {
        A[posA+k*(1+LDA)+i+j*LDA] += alpha*B[posB+i+j*LDB];
    }
}

void mblock::Add_Prod_Id(const double& alpha, const mblock &block1, const int &id_size, const int& bgn1, const int& bgn2, const char& id_pos, cudaStream_t stream) {
    assert(block1.sleft==1 || id_size==1);
    
    if (id_pos=='l') {
        dim3 block(8, 8, 8);
        dim3 grid( (block1.sleft-1)/block.x+1, (block1.sright-1)/block.y+1, (id_size-1)/block.z+1 );
        prodhelp_l<<<grid,block,0,stream>>>(alpha, mat, sleft, bgn2*sleft+bgn1, block1.mat, block1.sleft, 0, block1.sleft, block1.sright, id_size);
    }
    
    if (id_pos=='r') {
        dim3 block(8, 8, 8);
        dim3 grid( (block1.sleft-1)/block.x+1, (block1.sright-1)/block.y+1, (id_size-1)/block.z+1 );
        prodhelp_r<<<grid,block,0,stream>>>(alpha, mat, sleft, bgn2*sleft+bgn1, block1.mat, block1.sleft, 0, block1.sleft, block1.sright, id_size);
    }
}

ostream& operator<<(ostream& out, const mblock &block) {
    out << endl;
    cout << "L: {j=" << block.jleft << ",n=" << block.nleft << ",l=" << block.sleft << "}, R: {j=" << block.jright << ",n=" << block.nright << ",l=" << block.sright << "}" << endl;
	double* mymat = new double[block.sleft*block.sright];
    cublasGetMatrix(block.sleft, block.sright, sizeof(*(block.mat)), block.mat, block.sleft, mymat, block.sleft);
    for (int i = 0; i < block.sleft; ++i) {
		for (int j = 0; j < 1; j++) {
			cout << mymat[i+j*block.sleft] << " ";
		}
		// cout << endl;
	}
    cout << endl;
    delete [] mymat;
    return out;
}

void mblock::addto(const mblock &block, cudaStream_t stream) {
    if (checkplus(*this,block,'n')) {
        dim3 blockcu(32, 32);
        dim3 grid( (sleft-1)/blockcu.x+1, (sright-1)/blockcu.y+1 );
        matAdd<<<grid,blockcu,0,stream>>>(1.0, mat, sleft, 0, block.mat, sleft, 0, sleft, sright);
    }
}

mblock& mblock::operator +=(const mblock &block) {
    if (checkplus(*this,block,'n')) {
        dim3 blockcu(32, 32);
        dim3 grid( (sleft-1)/blockcu.x+1, (sright-1)/blockcu.y+1 );
        matAdd<<<grid,blockcu>>>(1.0, mat, sleft, 0, block.mat, sleft, 0, sleft, sright);
        // cudaDeviceSynchronize();
    }
    return *this;
}

mblockCPU::mblockCPU(const int& jleft, const int& jright, const int& sleft, const int& sright, const int& nleft, const int& nright) {
    this->jleft  = jleft; this->jright = jright;
    this->sleft  = sleft; this->sright = sright;
    this->nleft  = nleft; this->nright = nright;
    mat = new double[sleft*sright];
    // cudaMallocHost(&mat, sleft*sright*sizeof(double));
}

mblockCPU::mblockCPU(const mblockCPU& block):jleft(block.jleft),nleft(block.nleft),sleft(block.sleft),jright(block.jright),nright(block.nright),sright(block.sright) {
    mat = new double[sleft*sright];
    // cudaMallocHost(&mat, sleft*sright*sizeof(double));
    cblas_dcopy(sleft * sright, block.mat, 1, mat, 1);
}

mblockCPU::mblockCPU(ifstream& in, const char& flag) {
    in.read((char*) (&jleft),  sizeof(jleft));
    in.read((char*) (&jright), sizeof(jright));
    in.read((char*) (&nleft),  sizeof(nleft));
    in.read((char*) (&nright), sizeof(nright));
    in.read((char*) (&sleft),  sizeof(sleft));
    in.read((char*) (&sright), sizeof(sright));

    mat = new double[sleft*sright] ();
    // cudaMallocHost(&mat, sleft*sright*sizeof(double));
    if (flag=='n' || (flag=='s' && jleft < jright)) {
        in.read((char*) &mat[0], sleft*sright*sizeof(mat[0]));
    } else if (flag=='u' || (flag=='s' && jleft == jright)) {
        for (int i=0; i<sright; ++i) {
            in.read((char*) &mat[(1+sleft)*i], (sleft-i)*sizeof(mat[0]));
            for (int j=i+1; j<sleft; ++j) {
                mat[sleft*j+i]=mat[sleft*i+j];
            }
        }
    }
}

mblockCPU::~mblockCPU() {
    this->clear();
}

void mblockCPU::clear(){
    delete [] mat; mat=NULL;
    // cudaFreeHost(mat);
}

void mblockCPU::mconj(const double alpha){
    mkl_dimatcopy('c', 't', sleft, sright, alpha, mat, sleft, sright);

    int tmp;
	tmp = jright; jright = jleft; jleft = tmp;
    tmp=sright; sright=sleft; sleft=tmp;
    tmp=nright; nright=nleft; nleft=tmp;
}

ostream& operator<<(ostream& out, const mblockCPU &block) {
    out << endl;
    cout << "L: {j=" << block.jleft << ",n=" << block.nleft << ",l=" << block.sleft << "}, R: {j=" << block.jright << ",n=" << block.nright << ",l=" << block.sright << "}" << endl;
    cout << block.mat[0] << " ";
    cout << block.mat[block.sleft*block.sright-1] << " ";
    // for (int i = 0; i < block.sleft; ++i) {
	// 	for (int j = 0; j < block.sright; j++) {
	// 		cout << block.mat[i+j*block.sleft] << " ";
	// 	}
	// 	cout << endl;
	// }
    return out;
}