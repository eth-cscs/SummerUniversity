include ../make.inc

SOURCES = stats.cu  data.cu  operators.cu  linalg.cu    main.cu
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .cpp

all: main unit_tests

stats.o: stats.cu $(HEADERS)
	nvcc $(NVCC_FLAGS) -c stats.cu

data.o: data.cu $(HEADERS)
	nvcc $(NVCC_FLAGS) -c data.cu

operators.o: operators.cu $(HEADERS)
	nvcc $(NVCC_FLAGS) -c operators.cu

linalg.o: linalg.cu $(HEADERS)
	nvcc $(NVCC_FLAGS) -c linalg.cu

main.o: main.cu $(HEADERS)
	nvcc $(NVCC_FLAGS) -c main.cu

unit_tests.o: unit_tests.cu $(HEADERS)
	nvcc $(NVCC_FLAGS) -c unit_tests.cu

main: $(OBJ) main.o
	LIBRARY_PATH=${LD_LIBRARY_PATH} g++ $(GCC_FLAGS) $(LINK_FLAGS) $(OBJ) main.o -o main -lcudart -lcublas

unit_tests: data.o linalg.o unit_tests.o
	LIBRARY_PATH=${LD_LIBRARY_PATH} g++ $(GCC_FLAGS) $(LINK_FLAGS) unit_tests.o $(OBJ) -o unit_tests -lcudart -lcublas
#	./unit_tests                       # run with interactive session
	srun -Cgpu ./unit_tests                 # run without interactive session

clean:
	rm -f main
	rm -f unit_tests
	rm -f *.o
	rm -f *.i
	rm -f output.*
