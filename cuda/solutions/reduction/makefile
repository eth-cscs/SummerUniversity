include ../make.inc

all : reduction

reduction : reduction.cu
	nvcc $(NVCC_FLAGS) reduction.cu -o reduction $(LINK_FLAGS)

clean :
	rm -f reduction *.nvvp
