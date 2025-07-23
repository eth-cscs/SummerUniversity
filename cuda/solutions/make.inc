NVCC_FLAGS=-std=c++14 -arch=sm_90 -O3 -g -I../include
GCC_FLAGS=-std=c++14 -O3 -fopenmp -I../include
LINK_FLAGS=-lcublas -lcuda
