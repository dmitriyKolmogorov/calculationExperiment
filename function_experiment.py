import json
from time import time
from math import exp

from numba import jit, cuda, vectorize
import numpy as np

BATCH_SIZE_RANGE:range = range(50000, 1000000, 50000)
IMAGE_SIZE:int = 10
SAMPLE_SHAPE:int = 25
NUM_OF_EXPERIMENTS:int = 50
THREADS_PER_BLOCK:int = 256

@cuda.jit()
def cuda_version(values:np.ndarray) -> None:

    i = cuda.grid(1)

    if i < values.shape[0]:
        values[i] = 1 / (1 + exp(-values[i]))

cuda_version[1, 1](np.array([1.]))


def python_version(values:list) -> None:

    for i in range(len(values)):
        values[i] = 1 / (1 + exp(-values[i]))



def numpy_version(values:np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


@vectorize
def numba_vectorize_version(x:float) -> float:
    return 1 / (1 + np.exp(-x))

# compile function
numba_vectorize_version(np.array([255.]))


@jit
def numba_python_version(values:np.ndarray) -> None:
    for i in range(len(values)):
        values[i] = 1 / (1 + exp(-values[i]))

@jit
def numba_numpy_version(values:np.ndarray) -> None:
    return 1 / (1 + np.exp(-values))

numba_python_version(np.array([1.]))
numba_numpy_version(np.array([1.]))

func_names:tuple = ('python', 'numba_python', 'numba_numpy', 'numba_vectorize', 'numpy', 'cuda')

funcs:dict = {'numba_python':numba_python_version,
              'numba_numpy':numba_numpy_version,
              'numba_vectorize':numba_vectorize_version,
              'numpy':numpy_version,
              'cuda':cuda_version,
              'python':python_version}

def run() -> None:

    # start experiment
    records:dict = {func:dict() for func in func_names}

    for BATCH_SIZE in BATCH_SIZE_RANGE:

        print(BATCH_SIZE)

        samples:np.ndarray = np.random.random(BATCH_SIZE).astype(float)

        for func_name in func_names:
            time_sum:float = 0

            if func_name == 'cuda':
                blockspergrid = (BATCH_SIZE + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
                func = funcs[func_name][blockspergrid, THREADS_PER_BLOCK]
                samples:np.ndarray = cuda.to_device(samples)
            else:
                func = funcs[func_name]

            start = time()
            for i in range(NUM_OF_EXPERIMENTS):
                func(samples)

            records[func_name][BATCH_SIZE] = (time() - start) / NUM_OF_EXPERIMENTS

    with open('function_result.json', 'w') as json_file:
        json.dump(records, json_file, indent=4)

if __name__ == '__main__':
    run()