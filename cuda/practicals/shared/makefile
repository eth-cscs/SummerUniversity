include ../make.inc

CUDA_BENCHMARKS= blur blur_twice dot string_reverse hist
OPENMP_BENCHMARKS= blur_twice_omp

all : $(CUDA_BENCHMARKS) $(OPENMP_BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
blur : blur.cu
	nvcc $(NVCC_FLAGS) blur.cu -o blur $(LINK_FLAGS)

blur_twice : blur_twice.cu
	nvcc $(NVCC_FLAGS) blur_twice.cu -o blur_twice $(LINK_FLAGS)

dot : dot.cu
	nvcc $(NVCC_FLAGS) dot.cu -o dot $(LINK_FLAGS)

string_reverse : string_reverse.cu
	nvcc $(NVCC_FLAGS) string_reverse.cu -o string_reverse $(LINK_FLAGS)

hist : hist.cu
	nvcc $(NVCC_FLAGS) hist.cu -o hist $(LINK_FLAGS)

#----------------------------------------
# openmp benchmarks
#----------------------------------------
blur_twice_omp : blur_twice.cpp
	g++ $(GCC_FLAGS) blur_twice.cpp -o blur_twice_omp

clean :
	rm -f ${CUDA_BENCHMARKS}
	rm -f ${OPENMP_BENCHMARKS}
