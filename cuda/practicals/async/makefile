include ../make.inc

CUDA_BENCHMARKS=memcopy1 memcopy2 memcopy3

all : $(CUDA_BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
memcopy1 : memcopy1.cu
	nvcc $(NVCC_FLAGS) memcopy1.cu -o memcopy1 $(LINK_FLAGS)

memcopy2 : memcopy2.cu
	nvcc $(NVCC_FLAGS) memcopy2.cu -o memcopy2 $(LINK_FLAGS)

memcopy3 : memcopy3.cu
	nvcc $(NVCC_FLAGS) memcopy3.cu -o memcopy3 $(LINK_FLAGS)

clean :
	rm -f ${CUDA_BENCHMARKS}
