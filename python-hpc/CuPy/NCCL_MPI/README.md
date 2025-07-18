# NCCL All-Reduce using CuPy arrays

## Setting the toml environment

The following have to be set in the `pytorch.toml`: 

* `image = "/capstor/store/cscs/cscs/jupyter/pytorch/pytorch-summer-university-25.05-py3.sqsh"`

*  `mounts = ["<the current working directory containing the .py script>:/scratch"]`

The given example uses `mpi4py` to broadcast the unique id from the root rank.

### Running the example

The example can be easily executed from the current working directory, where 1 task is used per gpu (4 tasks per gpu):

```
srun --mpi=pmix -N <number of nodes> -n <number of tasks> -u --environment=$PWD/pytorch.toml python all_reduce_mpi4py_nccl.py
```
