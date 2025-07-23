include ../make.inc

BENCHMARKS= diffusion2d diffusion2d_mpi

all : $(BENCHMARKS)

#----------------------------------------
# diffusion2D benchmarks
#----------------------------------------
diffusion2d : diffusion2d.cu
	nvcc $(NVCC_FLAGS) diffusion2d.cu $(LINK_FLAGS) -o diffusion2d

# an aditional linking stage required for MPI
diffusion2d_mpi : diffusion2d_mpi.cu
	nvcc -c ${NVCC_FLAGS} diffusion2d_mpi.cu -I/user-environment/env/default/include/
	mpicxx diffusion2d_mpi.o -o diffusion2d_mpi -lcuda -lcudart -L/user-environment/env/default/lib64

clean :
	rm -f ${BENCHMARKS}
	rm -f *.o
	rm -f output.*
