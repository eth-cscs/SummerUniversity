import cupy as cp
import cupy.cuda.nccl as nccl

from cupy.cuda import Device 
from mpi4py import MPI

from timers import cpu_timer


global_seed = 42
GPUS_PER_NODE = 4


def init_comm(local_rank, global_rank, world_size, unique_id):
    with Device(local_rank): 
        comm = nccl.NcclCommunicator(world_size, unique_id, global_rank)
    
    return comm


def perform_send_recv(local_rank, global_rank, comm, world_size,
                      array_size, warmup=False):
    with Device(local_rank): 
        stream = cp.cuda.Stream()
        with stream:
            next_peer = (global_rank + 1) % world_size
            prev_peer = (global_rank - 1 + world_size) % world_size 
            engine = cp.random.default_rng(global_seed + global_rank) 
            x = engine.uniform(size=(array_size), dtype=cp.float64) 
            y = cp.empty_like(x) 
            stream.synchronize()

            # Warmup
            if warmup:
                nccl.groupStart()
                comm.send(x.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                          next_peer, stream.ptr)

                comm.recv(x.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                          prev_peer, stream.ptr)
                nccl.groupEnd()
                stream.synchronize()

            with cpu_timer() as timer:
                nccl.groupStart()
                comm.send(x.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                          next_peer, stream.ptr)
    
                comm.recv(x.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                          prev_peer, stream.ptr)
                nccl.groupEnd()

                stream.synchronize()
        bandwidth = (x.nbytes / 1024 ** 3) * 1000.0 /  timer.elapsed_time
        return bandwidth

if __name__ == '__main__':
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

    # Initialize the communicators here
    comm = init_comm(local_rank, rank, world_size, unique_id)
    
    for i in range(11):
        array_size = 1024 * 1024 * 2 ** i
        bandwidth = perform_send_recv(local_rank, rank, comm, world_size,
                                      array_size, warmup=(i == 0))
        bandwidth = mpi_comm.reduce(bandwidth, root=0, op=MPI.MIN)
        if rank == 0:
            print(f'Array size: {array_size // (1024 ** 2):5d}MB, bandwidth: {bandwidth:5.2f} GB/s')
