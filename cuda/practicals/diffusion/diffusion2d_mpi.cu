#include <cstdlib>

#include <fstream>
#include <iostream>
#include <vector>

#include <cuda.h>

#include <mpi.h>

#include "util.hpp"
#include "cuda_stream.hpp"

// 2D diffusion example with mpi
// the grid has a fixed width of nx=128
// the use specifies the height, ny, as a power of two
// note that nx and ny have 2 added to them to account for halos
//
// the domain decomposition is in the vertical
// ny is the height of the local sub-domain

void write_to_file(int nx, int ny, double* data, int mpi_size, int mpi_rank);

template <typename T>
void fill_gpu(T *v, T value, int n);

__global__
void diffusion(double *x0, double *x1, int nx, int ny, double dt) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x+1;
    auto j = threadIdx.y + blockIdx.y*blockDim.y+1;

    if (i<nx-1 && j<ny-1) {
        auto pos = i + j*nx;
        x1[pos] = x0[pos] + dt * (-4.*x0[pos]
                   + x0[pos-nx] + x0[pos+nx]
                   + x0[pos-1]  + x0[pos+1]);
    }
}

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    size_t pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    size_t nsteps = read_arg(argc, argv, 2, 100);

    // set domain size
    size_t nx = 128;
    size_t ny = 1 << pow;
    double dt = 0.1;

    // initialize MPI
    int mpi_rank, mpi_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    bool use_rdma = (nullptr!=std::getenv("MPICH_RDMA_ENABLED_CUDA"));

    // calculate global domain sizes
    if(ny%mpi_size) {
        std::cout << "error : global domain dimension " << ny
                  << "must be divisible by number of MPI ranks " << mpi_size
                  << std::endl;
        exit(1);
    }
    else if(mpi_rank==0) {
        std::cout << "------------------------------------------\n";
        std::cout << mpi_size << " MPI ranks, "
                  << (use_rdma? "with RDMA\n": "no RDMA\n");
        std::cout << nx << "x" << ny
                  << " : " << nx << "x" << ny/mpi_size << " per rank"
                  << " for " << nsteps << " time steps"
                  << " (" << nx*ny << " grid points)"
                  << std::endl;
    }
    ny /= mpi_size;

    // adjust dimensions for halo
    nx += 2;
    ny += 2;

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = nx*ny;
    double *x_host = malloc_pinned<double>(buffer_size);
    double *x0     = malloc_device<double>(buffer_size);
    double *x1     = malloc_device<double>(buffer_size);

    // set initial conditions of 0 everywhere
    fill_gpu(x0, 0., buffer_size);
    fill_gpu(x1, 0., buffer_size);

    // set boundary conditions of 1 on south border
    if(mpi_rank==0) { // south boundary
        fill_gpu(x0, 1., nx);
        fill_gpu(x1, 1., nx);
    }
    if(mpi_rank==mpi_size-1) { // north boundary
        fill_gpu(x0+nx*(ny-1), 1., nx);
        fill_gpu(x1+nx*(ny-1), 1., nx);
    }

    cuda_stream stream;
    cuda_stream copy_stream{};
    auto start_event = stream.enqueue_event();

    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
            (nx-3)/block_dim.x+1,
            (ny-3)/block_dim.y+1);

    MPI_Status status_north;
    MPI_Status status_south;

    auto recv_buffer = malloc_pinned<double>(nx);
    auto send_buffer = malloc_pinned<double>(nx);

    // time stepping loop
    for(auto step=0; step<nsteps; ++step) {

        // perform halo exchange
        // x0(:, 0)    <- south
        // x0(:, 1)    -> south
        // x0(:, ny-1) <- north
        // x0(:, ny-2) -> north
        if (use_rdma) {
            // TODO: fill in g2g communication
        }
        else {
            if (mpi_rank>0) {
                 copy_to_host(x0+nx, send_buffer, nx);
                 MPI_Sendrecv(send_buffer, nx, MPI_DOUBLE,
                            mpi_rank-1, 0,
                            recv_buffer, nx, MPI_DOUBLE,
                            mpi_rank-1, 1,
                            MPI_COMM_WORLD, &status_south);
                 copy_to_device(recv_buffer, x0, nx);
            }
            if (mpi_rank<mpi_size-1) {
                 copy_to_host(x0+(ny-2)*nx, send_buffer, nx);
                 MPI_Sendrecv(send_buffer, nx, MPI_DOUBLE,
                            mpi_rank+1, 1,
                            recv_buffer, nx, MPI_DOUBLE,
                            mpi_rank+1, 0,
                            MPI_COMM_WORLD, &status_north);
                 copy_to_device(recv_buffer, x0+(ny-1)*nx, nx);
            }
        }
        diffusion<<<grid_dim, block_dim>>>(x0, x1, nx, ny, dt);

        std::swap(x0, x1);
    }
    auto stop_event = stream.enqueue_event();
    stop_event.wait();

    copy_to_host<double>(x0, x_host, buffer_size);

    double time = stop_event.time_since(start_event);

    if(mpi_rank==0) {
        std::cout << "time " << time << " s, "
                  << nsteps*(nx-2)*(ny-2)*mpi_size / time << " points/second"
                  << std::endl;
    }
    write_to_file(nx, ny, x_host, mpi_size, mpi_rank);

    MPI_Finalize();

    return 0;
}

template <typename T>
__global__
void fill(T *v, T value, int n) {
    int tid  = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid<n) {
        v[tid] = value;
    }
}

template <typename T>
void fill_gpu(T *v, T value, int n) {
    auto block_dim = 192ul;
    auto grid_dim = n/block_dim + (n%block_dim ? 1 : 0);

    fill<T><<<grid_dim, block_dim>>>(v, value, n);
}

void write_to_file(int nx, int ny, double* data, int mpi_size, int mpi_rank) {
    // collect the global solution to the root rank
    auto block_size = nx*(ny-2); // discard first and last rows
    std::vector<double> data_global(mpi_size*block_size);
    MPI_Gather(data+nx,          block_size, MPI_DOUBLE,
               &data_global[0], block_size, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if(mpi_rank==0) {
        FILE* output = fopen("output.bin", "w");
        fwrite(&data_global[0], sizeof(double), mpi_size* nx * (ny), output);
        fclose(output);

        std::ofstream fid("output.bov");
        fid << "TIME: 0.0" << std::endl;
        fid << "DATA_FILE: output.bin" << std::endl;
        fid << "DATA_SIZE: " << nx << " " << mpi_size*(ny) << " 1" << std::endl;;
        fid << "DATA_FORMAT: DOUBLE" << std::endl;
        fid << "VARIABLE: phi" << std::endl;
        fid << "DATA_ENDIAN: LITTLE" << std::endl;
        fid << "CENTERING: nodal" << std::endl;
        fid << "BRICK_SIZE: 1.0 1.0 1.0" << std::endl;
    }
}
