import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from my_linear_lib import is_matrix_orthogonal, gs

SIZE = 4
SCALE = 2

rnd = np.random.normal(size=SIZE ** 2)
rnd = rnd.reshape((SIZE, SIZE))

rnd = np.random.normal(size=SIZE **2).reshape((SIZE, SIZE))
rnd = rnd * SCALE
my_Q , my_R = gs(rnd)

Q, R = np.linalg.qr(rnd, "complete")
print(np.allclose(my_Q, Q))

