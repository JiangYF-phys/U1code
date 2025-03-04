CXX=nvcc
CXXFLAGS= -I${MKLROOT}/include -DMKL_ILP64 -std=c++11 -O3 -g # -Xcompiler  -rdynamic -G # -lineinfo # -traceback# -DNDEBUG 

ALL_LIBS= -L${MKLROOT}/lib/intel64  -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lcublas -lcusolver
BIN=dmrg

OBJ=mblock.o reducematrix.o hamiltonian.o repmap.o extend.o wavefunc.o truncation.o lanczos.o lattice.o global.o DMRG.o measure.o main.o


all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(ALL_LIBS) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	
%.o : %.cu
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o
	rm $(BIN)
