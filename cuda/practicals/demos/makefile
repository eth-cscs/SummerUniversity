include ../make.inc

CUDA_BENCHMARKS=hello bandwidth

BENCHMARKS=$(CUDA_BENCHMARKS)

all : $(BENCHMARKS)

hello : hello.cu
	nvcc $(NVCC_FLAGS) hello.cu -o hello $(LINK_FLAGS)

bandwidth : bandwidth.cu
	nvcc $(NVCC_FLAGS) bandwidth.cu -o bandwidth $(LINK_FLAGS)

clean :
	rm -f $(BENCHMARKS) *.nvvp
