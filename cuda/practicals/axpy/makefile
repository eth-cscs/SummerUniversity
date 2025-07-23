include ../make.inc

all : axpy axpy_omp newton

axpy : axpy.cu
	nvcc $(NVCC_FLAGS) axpy.cu -o axpy $(LINK_FLAGS)

axpy_omp : axpy_omp.cpp
	#g++ ${GCC_FLAGS} axpy_omp.cpp -o axpy_omp -march=core-avx2
	g++ ${GCC_FLAGS} axpy_omp.cpp -o axpy_omp -march=native
	#g++ ${GCC_FLAGS} axpy_omp.cpp -o axpy_omp

newton : newton.cu
	nvcc $(NVCC_FLAGS) newton.cu -o newton $(LINK_FLAGS)

clean :
	rm -f axpy axpy_omp newton *.nvvp
