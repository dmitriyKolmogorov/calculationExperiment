import json
from time import time

from numba import jit, cuda, vectorize
import numpy as np

BATCH_SIZE_RANGE:range = range(50, 1000, 50)
IMAGE_SIZE:int = 10
SAMPLE_SHAPE:int = 25
NUM_OF_EXPERIMENTS:int = 25
THREADS_PER_BLOCK:int = 256

@cuda.jit()
def cuda_version(images:np.ndarray) -> None:

    i1, i2, i3 = cuda.grid(3)

    if i1 < images.shape[0] and i2 < images[i1].shape[0] and i3 < images[i1, i2].shape[0]:
        for i4 in range(images.shape[3]):
            images[i1][i2][i3][i4] /= 255

cuda_version[1, 1](np.random.random((1, 1, 1, 1)))


def python_version(images:list) -> None:

    for i1 in range(len(images)):
        for i2 in range(len(images[i1])):
            for i3 in range(len(images[i1][i2])):
                for i4 in range(len(images[i1][i2][i3])):
                    images[i1][i2][i3][i4] /= 255


def numpy_version(images:np.ndarray) -> np.ndarray:
    return np.divide(images, 255)


@vectorize
def numba_vectorize_version(x:float) -> float:
    return x / 255

# compile function
numba_vectorize_version(255)


@jit
def numba_python_version(images:np.ndarray) -> None:
    d1, d2, d3, d4 = images.shape

    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                for p in range(d4):
                    images[i][j][k][p] /= 255

@jit
def numba_numpy_version(images:np.ndarray) -> None:
    return np.divide(images, 255)

numba_python_version(np.random.random((1, 1, 1, 1)))
numba_numpy_version(np.random.random((1, 1, 1, 1)))

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

        samples:np.ndarray = np.random.randint(low=0, high=255, size=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(float)

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

    with open('image_result.json', 'w') as json_file:
        json.dump(records, json_file, indent=4)

if __name__ == '__main__':
    run()