import json
from time import time

from numba import cuda, jit, vectorize
import numpy as np

BATCH_SIZE_RANGE:range = range(5000, 200001, 5000)
SAMPLE_SHAPE:int = 15
NUM_OF_EXPERIMENTS:int = 25
THREADS_PER_BLOCK:int = 256
LEARNING_RATE:float = 0.05


@cuda.jit('(f8[:], f8[:, :], f8[:])')
def cuda_version(weights:np.ndarray, 
                 samples:np.ndarray, 
                 target:np.ndarray) -> None:

    i:int = cuda.grid(1)

    if i < samples.shape[0]:
        pred:float = 0

        for j in range(weights.shape[0]):
            pred = pred +  weights[j] * samples[i, j]

        for j in range(weights.shape[0]):
            weights[j] = weights[j] - LEARNING_RATE * samples[i, j] * (target[i] - pred)

# compile cuda function first time
cuda_version[1, 1](np.zeros(1).astype(float),
                   np.zeros((1, 1)).astype(float),
                   np.zeros(1).astype(float))


def python_version(weights:list,
                   samples:list,
                   target:list) -> None:

    for i in range(len(samples)):

        sample:list = samples[i]
        value:float = target[i]

        predicted:float = 0

        for j in range(len(weights)):
            predicted += weights[j] * sample[j]

        for j in range(len(weights)):
            weights[j] -= sample[j] * LEARNING_RATE * (value - predicted)


def numpy_version(weights:np.ndarray, 
                  samples:np.ndarray, 
                  target:np.ndarray) -> None:

    dw:np.ndarray = np.reshape((target - np.sum(weights * samples)), (-1, 1))

    weights -= LEARNING_RATE * np.sum(samples * dw)


@jit('(f8[:], f8[:, :], f8[:])')
def numba_python_version(weights:np.ndarray, 
                  samples:np.ndarray, 
                  target:np.ndarray) -> None:

    for i in range(samples.shape[0]):

        sample:list = samples[i]

        predicted:float = 0

        for j in range(weights.shape[0]):
            predicted += weights[j] * sample[j]

        for j in range(weights.shape[0]):
            weights[j] -= sample[j] * LEARNING_RATE * (target[i] - predicted)


@jit('(f8[:], f8[:, :], f8[:])')
def numba_numpy_version(weights:np.ndarray, 
                        samples:np.ndarray, 
                        target:np.ndarray) -> None:

    dw:np.ndarray = np.reshape((target - np.sum(weights * samples)), (-1, 1))

    weights -= LEARNING_RATE * np.sum(samples * dw)


numba_python_version(np.zeros(1).astype(float),
                     np.zeros((1, 1)).astype(float),
                     np.zeros(1).astype(float))

numba_numpy_version(np.zeros(1).astype(float),
                    np.zeros((1, 1)).astype(float),
                    np.zeros(1).astype(float))

func_names:tuple = ('python', 'numba_python', 'numba_numpy', 'numpy', 'cuda')

funcs:dict = {'numba_python':numba_python_version,
              'numba_numpy':numba_numpy_version,
              'numpy':numpy_version,
              'cuda':cuda_version,
              'python':python_version}

def run() -> None:

    # start experiment
    records:dict = {func:dict() for func in func_names}

    for BATCH_SIZE in BATCH_SIZE_RANGE:

        print(BATCH_SIZE)

        weights:np.ndarray = np.random.random(SAMPLE_SHAPE)
        samples:np.ndarray = np.random.random((BATCH_SIZE, SAMPLE_SHAPE))
        target:np.ndarray = np.random.random(BATCH_SIZE)

        for func_name in func_names:
            time_sum:float = 0

            if func_name == 'cuda':
                blockspergrid = (BATCH_SIZE + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
                func = funcs[func_name][blockspergrid, THREADS_PER_BLOCK]
                weights:np.ndarray = cuda.to_device(weights)
                samples:np.ndarray = cuda.to_device(samples)
                target:np.ndarray = cuda.to_device(target)
            else:
                func = funcs[func_name]

            start:float = time()
            for i in range(NUM_OF_EXPERIMENTS):
                func(weights, samples, target)

            records[func_name][BATCH_SIZE] = (time() - start) / NUM_OF_EXPERIMENTS

    with open('sgd_result.json', 'w') as json_file:
        json.dump(records, json_file, indent=4)

if __name__ == '__main__':
    run()