include ../make.inc

CUDA_BENCHMARKS=explicit managed sort

all : $(CUDA_BENCHMARKS)

explicit : explicit.cu util.hpp
	nvcc $(NVCC_FLAGS) explicit.cu -o explicit $(LINK_FLAGS)

managed : managed.cu util.hpp
	nvcc $(NVCC_FLAGS) managed.cu -o managed $(LINK_FLAGS)

# this performance test must be compiled without debug flags (-g -G)
# otherwise performance suffers a lot
sort : sort.cu util.hpp
	nvcc $(NVCC_FLAGS) sort.cu -o sort $(LINK_FLAGS)

clean :
	rm -f $(CUDA_BENCHMARKS) *.nvvp
