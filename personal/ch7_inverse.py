import numpy as np

Q1 = np.array([
    [1, -1],
    [1, 1],
    [2, 3]
]) / np.sqrt(2)

Q, R = np.linalg.qr(Q1)

print(Q,"\n\n", R, "\n\n", Q @ R * np.sqrt(2))