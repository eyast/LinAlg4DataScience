import numpy as np

# Example: 3 vectors in R^3
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([1, 0, 1])

A = np.vstack([v1, v2, v3])
print(A)
rank = np.linalg.matrix_rank(A)
print("Rank:", rank)