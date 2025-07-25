import numpy as np
import cupy as cp
import cupyx

from timers import cupy_timer, cpu_timer


CYCLES = 10

print('Running benchmark for D -> H Pageable')

for i in range(11):
    device_array = cp.zeros(shape=1024 * 1024 * 2**i, dtype=cp.byte)
    host_array = np.zeros(shape=1024 * 1024 * 2**i, dtype=np.byte)
    
    # Warmup
    cp.asnumpy(device_array, out=host_array)
    
    with cupy_timer() as timer:
        for c in range(CYCLES):
            cp.asnumpy(device_array, out=host_array, blocking=False)

    bandwidth = device_array.nbytes * CYCLES / (1024 ** 3) / timer.elapsed_time * 1000
    print(f'{2**i:4d}MB -> {bandwidth:4.2f} GB/s')

print('Running benchmark for D -> H Pinned')

for i in range(11):
    device_array = cp.zeros(shape=1024 * 1024 * 2 ** i, dtype=cp.byte)
    host_array = []
    for j in range(4):
        host_array.append(cupyx.empty_pinned(shape=1024 * 1024 * 2 ** i, dtype=np.byte))

    streams = [None] * 4 
    
    # Warmup
    cp.asnumpy(device_array, out=host_array[0])
    
    with cpu_timer() as timer:
        for c in range(CYCLES):
            for l in range(4):
                with cp.cuda.Stream() as s:
                    cp.asnumpy(device_array, stream=s, out=host_array[l], blocking=False)
                    streams[l] = s

            for s in streams:
                s.synchronize()

    bandwidth = 4 * device_array.nbytes * CYCLES / (1024 ** 3) / timer.elapsed_time * 1000
    print(f'{2**i:4d}MB -> {bandwidth:4.2f} GB/s')

print('Running benchmark for H -> D Pageable')

for i in range(11):
    host_array = np.zeros(shape=1024 * 1024 * 2**i, dtype=np.byte)
    device_array = cp.zeros(shape=1024 * 1024 * 2**i, dtype=cp.byte)
    
    # Warmup
    device_array.data.copy_from_host(host_array.ctypes.data, host_array.nbytes)
    
    with cupy_timer() as timer:
        for c in range(CYCLES):
            device_array.data.copy_from_host(host_array.ctypes.data, host_array.nbytes)

    bandwidth = device_array.nbytes * CYCLES / (1024 ** 3) / timer.elapsed_time * 1000
    print(f'{2**i:4d}MB -> {bandwidth:4.2f} GB/s')

print('Running benchmark for H -> D Pinned')

for i in range(11):
    device_array = cp.zeros(shape=1024 * 1024 * 2 ** i, dtype=cp.byte)
    host_array = cupyx.zeros_pinned(shape=1024 * 1024 * 2 ** i, dtype=np.byte)
    
    # Warmup
    device_array.data.copy_from_host(host_array.ctypes.data, host_array.nbytes)
    
    with cupy_timer() as timer:
        for c in range(CYCLES):
            device_array.data.copy_from_host(host_array.ctypes.data, host_array.nbytes)

    bandwidth = device_array.nbytes * CYCLES / (1024 ** 3) / timer.elapsed_time * 1000
    print(f'{2**i:4d}MB -> {bandwidth:4.2f} GB/s')

print('Running benchmark for D0 -> D0')

for i in range(13):
    device_array = cp.zeros(shape=1024 * 1024 * 2 ** i, dtype=cp.byte)
    device_array2 = cp.zeros(shape=1024 * 1024 * 2 ** i, dtype=cp.byte)
    
    # Warmup
    device_array2[:] = device_array[:] 
    cp.cuda.get_current_stream().synchronize()
     
    with cupy_timer() as timer:
        for c in range(CYCLES):
            device_array2[:] = device_array[:]

    bandwidth = device_array.nbytes * CYCLES / (1024 ** 3) / timer.elapsed_time * 1000
    print(f'{2**i:4d}MB -> {bandwidth:4.2f} GB/s')

print('Running benchmark for D0 -> D1')

for i in range(13):
    with cp.cuda.Device(0):
        device_array = cp.zeros(shape=1024 * 1024 * 2 ** i, dtype=cp.byte)
    with cp.cuda.Device(1):
        device_array2 = cp.zeros(shape=1024 * 1024 * 2 ** i, dtype=cp.byte)
    
    # Warmup
    device_array2[:] = device_array[:] 
    cp.cuda.get_current_stream().synchronize()
     
    with cupy_timer() as timer:
        for c in range(CYCLES):
            device_array2[:] = device_array[:]

    bandwidth = device_array.nbytes * CYCLES / (1024 ** 3) / timer.elapsed_time * 1000
    print(f'{2**i:4d}MB -> {bandwidth:4.2f} GB/s')
