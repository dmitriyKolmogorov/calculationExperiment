import sys
import json

filename:str = sys.argv[1]

if filename == 'sgd': 
    filename:str = 'sgd_result' 
elif filename == 'image':
    filename:str = 'image_result'
else:
    filename:str = 'function_result'

    
import numpy as np
from matplotlib import pyplot as plt

labels:dict = {'numba_python':'Numba (using pure Python)',
               'numba_numpy':'Numba (using NumPy)',
               'numba_vectorize':'Numba (vectorized)',
               'numpy':'NumPy',
               'cuda':'CUDA',
               'python':'Pure Python'}

with open(f'{filename}.json') as json_file:
    data:dict = json.load(json_file)

for func in data:
    x:np.ndarray = np.array(list(data[func].keys()))
    y:np.ndarray = np.array(list(data[func].values()))

    y:np.ndarray = np.log10(y)

    plt.plot(x, y, label=labels[func])

plt.xlabel('Number of samples in input') 
plt.ylim([-6, 1])
plt.ylabel(r'$log_{10} t$')
plt.title('How execution time  $(t)$ depends on number of samples $(N)$\n in input for different tools')
plt.xticks(x, [str(v) for v in x], rotation=90)
plt.legend(loc='lower center', fontsize=16, ncol=3)
plt.grid(True)

plt.savefig(filename + '.png')

plt.show()
