# Running the NCCL All-Reduce example

The given example makes the assumption of a shared file system, so that all the processes have access to the unique id.
Therefore, the following should be executed from the `$SCRATCH` directory which points to `/capstor`.

## Setting the toml environment

The following have to be set in the `pytorch.toml`: 

* `image = "<path to a PyTorch NGC based image>"`

*  `mounts = ["<the current working directory containing the .py script>:/scratch"]`

## Running the example

The example can be easily executed from the current working directory:

```
srun -N <number of nodes> -u --environment=$PWD/pytorch.toml python all_reduce.py
```
 
