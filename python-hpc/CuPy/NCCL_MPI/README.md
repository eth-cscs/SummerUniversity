# Running the NCCL All-Reduce examples

## Run the example using a shared file for initialization

The given example makes the assumption of a shared file system, so that all the processes have access to the unique id.
Therefore, the following should be executed from the `$SCRATCH` directory which points to `/capstor`.

### Setting the toml environment

The following have to be set in the `pytorch.toml`: 

* `image = "<path to a PyTorch NGC based image>"`

*  `mounts = ["<the current working directory containing the .py script>:/scratch"]`

### Running the example

The example can be easily executed from the current working directory, where 1 task is used per node:

```
srun -N <number of nodes> -u --environment=$PWD/pytorch.toml python all_reduce_nccl.py
```
 
## Run the example using `mpi4py` for initialization

The given example uses `mpi4py` to broadcast the unique id from the root rank.

### Setting the toml environment

A working image can be created starting from a Pytorch NGC-based one and just installing `mpi4py` with `pip`.
The corresponding environment file `pytorch_mpi4py.toml`: 

* `image = "<path to a PyTorch NGC (with mpi4py) based image>"`

*  `mounts = ["<the current working directory containing the .py script>:/scratch"]`

### Running the example

The example can be easily executed from the current working directory, where 1 task is used per gpu (4 tasks per gpu):

```
srun --mpi=pmix -N <number of nodes> -n <number of tasks> -u --environment=$PWD/pytorch_mpi4py.toml python all_reduce_mpi4py_nccl.py
```
 
