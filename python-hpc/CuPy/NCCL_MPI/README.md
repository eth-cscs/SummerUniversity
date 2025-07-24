# NCCL All-Reduce using CuPy arrays

## Setting the toml environment

The following have to be set in the `pytorch.toml`: 

* `image = "/capstor/store/cscs/cscs/jupyter/pytorch/pytorch-summer-university-25.05-py3.sqsh"`

*  `mounts = ["<the current working directory containing the .py script>:/scratch"]`

The given example uses `mpi4py` to broadcast the unique id from the root rank.

### Running the example

The example can be easily executed from the current working directory, where 1 task is used per gpu (4 tasks per node):

```
srun --mpi=pmix -N <number of nodes> -n <number of tasks> -u --environment=$PWD/pytorch.toml python all_reduce_mpi4py_nccl.py
```

### Exercise

Implement a [sedrecv](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv) using CuPy arrays and NCCL. 

Use the above `sendrecv` to create a chain of `n` processes where each process `i`, receives data from it's left neighbor process `i-1` and sends data to it's right neighbor process `i+1` simultaneously. 

For process `0`,  use process `n-1` as it's left neighbor, while for process `n-1` use process `0` as it's right neighbor.

By measuring the time needed for the transfers to complete, the network bandwidth can be calculated.
