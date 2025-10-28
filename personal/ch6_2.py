import numpy as np
from numpy.typing import NDArray
import time


def find_distance(mat1: NDArray, mat2: NDArray) -> float:
    assert mat1.shape == mat2.shape, "Shape mismatch"
    mat3 = mat2 - mat1
    prod = mat3 @ mat3.T
    trace = np.trace(prod)
    norm = np.sqrt(trace)
    return norm

NUM = 25

rnd_1 = np.random.normal(size=NUM **2).reshape((NUM, NUM))
rnd_2 = np.random.normal(size=NUM **2).reshape((NUM, NUM))
s = 1
iters = 0
dist = find_distance(rnd_1, rnd_2)
print(f"Starting at dist: {dist}")
while dist > 1:
    rnd_1 *= s
    rnd_2 *= s
    dist = find_distance(rnd_1, rnd_2)
    s *= 0.9
    iters += 1

print(f"After {iters} iterations. Value of s: {s}")
time.sleep(1)