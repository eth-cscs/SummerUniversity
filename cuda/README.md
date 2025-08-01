## CUDA Lesson Plan

### Day 1

#### Introduction

Understand motivations behind using GPUs for HPC. Key architectural distinctions between CPU & GPU. Get started with connecting with the HPC cluster at CSCS.

`01_introduction.pdf`

`02_porting.pdf`

#### CUDA API

Learn the programming model, common GPU libraries, understand GPU memory management with practical exercises.

`03_runtime_api.pdf`

Exercises under `practicals/api` folder.

#### CUDA Kernels & Threads

Getting started with writing custom GPU kernels, understanding concepts of CUDA threads, blocks and grids with practical exercises

`04_kernels.pdf`

Exercises under `practicals/axpy` folder.

### Day 2

#### Kernels & Threads

Review, continuation and wrap-up

`04_kernels.pdf`

#### Shared Memory and Block Syncronization

Learn using cooperating thread blocks for more advanced kernels. Understand concepts such as race conditions, thread synchronization, atomics with practical exercises.

`05_shared.pdf`

Exercises under `practicals/shared` folder.

#### CUDA 2D

Learn to use the CUDA api for data in 2D arrays. Useful for many common scientific applications.

`06_cuda2d.pdf`

Exercises under `practicals/diffusion` folder.

### Day 3

#### 2D Diffusion Miniapp

Understand implementing a real-world numerical simulation using a toy mini-app. Leverage previous concepts to implement working GPU code, and compare with a CPU version. The same example would be extended for future lessons on OpenACC as well.

`07_miniapp_intro.pdf`

`08_miniapp_exercise.pdf`

Coding exercises in the `practicals/miniapp` folder.

#### Advanced GPU Concepts

Asynchronous operations for concurrency, and using GPUs in distributed computing. We'll try to cover as much of this as possible depending on the time, interest from the participants and other considerations in lectures, but this extra content along with practical examples will be made for motivated learners.

`bonus/09_async.pdf`
Exercises under `practicals/async` folder.

`bonus/10_advanced_porting.pdf`
Optional: Exercises under `practicals/trees` folder.

`bonus/11_cuda_mpi.pdf`
Exercises under `practicals/diffusion` folder.

`bonus/12_reduction.pdf`
Exercises under `practicals/reduction` folder.

#### NOTE: Solutions would be uploaded in the end of the day in the same repo in the `solutions/` folder.
