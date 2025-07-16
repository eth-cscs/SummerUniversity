import os
import multiprocessing 
import pickle
import time

import cupy as cp
import cupy.cuda.nccl as nccl
from cupy.cuda import Device 

# This is needed in order to work correctly with multiple processes
multiprocessing.set_start_method('spawn', force=True)

global_seed = 42

def perform_all_reduce(local_rank, global_rank, unique_id, total_ranks, array_size):
    with Device(local_rank): 
        stream = cp.cuda.Stream()
        with stream:
            engine = cp.random.default_rng(global_seed + global_rank) 
            x = engine.uniform(size=(array_size), dtype=cp.float64) 
            y = cp.empty_like(x) 
            comm = nccl.NcclCommunicator(total_ranks, unique_id, global_rank)
            comm.allReduce(x.data.ptr, y.data.ptr, cp.size(x), nccl.NCCL_FLOAT64,
                           nccl.NCCL_SUM, stream.ptr)

        stream.synchronize()
        print(f'Rank: {global_rank} -> {y.mean()}')


if __name__ == '__main__':
    num_devices_per_node = 4
    array_size = 10000000
    procs = [] 
    node_rank = int(os.environ['SLURM_PROCID'])
    world_size = num_devices_per_node * int(os.environ['SLURM_NNODES'])

    # Create a NCCL unique ID
    # This is not the best way to communicate the unique ID, it's better
    # to use something like MPI to do so.
    if node_rank == 0:
        unique_id = nccl.get_unique_id();
        with open(f'uniqueid_{os.environ["SLURM_JOBID"]}', 'wb') as file:
            pickle.dump(unique_id, file)

        with open(f'uniqueid_{os.environ["SLURM_JOBID"]}.lock', 'w') as file:
            pass  
    else:
        for i in range(10):
            if not os.path.exists(f'uniqueid_{os.environ["SLURM_JOBID"]}.lock'):
                time.sleep(0.5)

        with open(f'uniqueid_{os.environ["SLURM_JOBID"]}', 'rb') as file:
            unique_id = pickle.load(file)
        
    for i in range(num_devices_per_node):
        local_rank = i
        global_rank = node_rank * num_devices_per_node + i
        t = multiprocessing.Process(
            target=perform_all_reduce,
            args=(local_rank, global_rank, unique_id, world_size, array_size)
            )
        procs.append(t)
     
    for i in range(num_devices_per_node):
        procs[i].start()

    for i in range(num_devices_per_node):
        procs[i].join()
