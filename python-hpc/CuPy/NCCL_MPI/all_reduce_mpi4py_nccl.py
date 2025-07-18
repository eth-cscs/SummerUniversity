import cupy as cp
import cupy.cuda.nccl as nccl

from cupy.cuda import Device 
from mpi4py import MPI

from timers import cupy_timer


global_seed = 42
GPUS_PER_NODE = 4


def perform_all_reduce(local_rank, global_rank, unique_id, world_size, array_size):
    with Device(local_rank): 
        stream = cp.cuda.Stream()
        with stream:
            engine = cp.random.default_rng(global_seed + global_rank) 
            x = engine.uniform(size=(array_size), dtype=cp.float64) 
            y = cp.empty_like(x) 
            stream.synchronize()
            comm = nccl.NcclCommunicator(world_size, unique_id, global_rank)

            # Warmup
            comm.allReduce(x.data.ptr, y.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                           nccl.NCCL_SUM, stream.ptr)
            stream.synchronize()

            with cupy_timer() as timer:
                comm.allReduce(x.data.ptr, y.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                               nccl.NCCL_SUM, stream.ptr)

        stream.synchronize()
        bandwidth = (x.nbytes / 1024 ** 3) * 1000.0 /  timer.elapsed_time * ( 2 * (world_size -1 ) / world_size)
        return bandwidth #print(f'Rank: {global_rank} -> {y.mean()}: {bandwidth} GB/s')


if __name__ == '__main__':
    array_size = 1_000_000_000
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()

    # Get the unique_id in rank 0 and broadcast it to other ranks
    if rank == 0:
        unique_id = nccl.get_unique_id();
    else:
        unique_id = None 
        
    unique_id = mpi_comm.bcast(unique_id, root=0)

    local_rank = rank % GPUS_PER_NODE
    bandwidth = perform_all_reduce(local_rank, rank, unique_id, world_size, array_size)
    bandwidth = mpi_comm.reduce(bandwidth, root=0)

    if rank == 0:
        print(f'Array size: {array_size}, bandwidth: {bandwidth / world_size:5.2f} GB/s')

