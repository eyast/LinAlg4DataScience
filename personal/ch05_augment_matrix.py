import numpy as np
import matplotlib.pyplot as plt
import time

NUM_SCALARS = 50
NUM_EXP = 10

results = np.zeros((NUM_SCALARS, NUM_EXP))
for s in range(NUM_SCALARS):
    for x in range(NUM_EXP):
        rnd = np.random.normal(size=100).reshape((10, 10))
        matrix = rnd * s
        square = matrix @ matrix.T
        trace = np.trace(square)
        norm = np.sqrt(trace)
        results[s,x] = norm

means = np.mean(results, axis=1)
plt.scatter(
    list(range(NUM_SCALARS)), means
)
plt.show()
time.sleep()