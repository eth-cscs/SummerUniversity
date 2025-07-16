import cupy as cp
import cupy.cuda.nccl as nccl

from mpi4py import MPI
from cupy.cuda import Device 


global_seed = 42
GPUS_PER_NODE = 4


if __name__ == '__main__':
    array_size = 10000000
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()

    local_rank = rank % GPUS_PER_NODE

    with Device(local_rank): 
        stream = cp.cuda.Stream()
        with stream:
            engine = cp.random.default_rng(global_seed + rank) 
            x = engine.uniform(size=(array_size), dtype=cp.float64) 
            y = cp.empty_like(x) 
            stream.synchronize()
            mpi_comm.Allreduce(x, y)

        stream.synchronize()
        print(f'Rank: {rank} -> {y.mean()}')
