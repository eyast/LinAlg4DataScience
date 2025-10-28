import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import time

def matrix_norm(matrix: NDArray) -> float:
    r"""
    Calculates the Frobenius norm.
    """
    square = matrix @ matrix.T
    trace = np.trace(square)
    norm = np.sqrt(trace)
    return norm

M = 10
r = 6
N = 5

rnd_1 = np.random.normal(size = M * r).reshape((M, r))
rnd_2 = np.random.normal(size = r * N).reshape((r, N))

out = rnd_1 @ rnd_2
rank = np.linalg.matrix_rank(out)
print(f"Rank is: {rank}")