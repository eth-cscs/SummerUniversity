import cupy as cp

from contextlib import contextmanager
from timeit import default_timer


class Timer():
    def __init__(self):
        self._elapsed_time = 0.0

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, val):
        self._elapsed_time = val


@contextmanager
def cupy_timer(log=False):
    timer = Timer()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    yield timer
    end.record()
    end.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start, end)
    timer.elapsed_time = elapsed_time
    if log:
        print(f'Elapsed time: {elapsed_time} ms')


@contextmanager
def cpu_timer(log=False):
    timer = Timer()
    start = default_timer()
    yield timer
    end = default_timer()
    timer.elapsed_time = (end - start) * 1000
    if log:
        print(f'Elapsed time: {(end - start) * 1000} ms')
