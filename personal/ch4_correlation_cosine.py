import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def correlation(vec1: NDArray, vec2: NDArray) -> float:
    """
    Chapter 4 exercise - Correlation
        
        Args:
    vec1 (np.array): vector
    vec2 (np.array): vector
    
        Returns:
    Cosine Similarity (float)
    """
    assert vec1.shape == vec2.shape, "Inconsistent shapes provided"
    vec1_centered = vec1 - np.mean(vec1)
    vec2_centered = vec2 - np.mean(vec2)
    return np.dot(
                vec1_centered.T,
                vec2_centered) / (np.linalg.norm(vec1_centered) * np.linalg.norm(vec2_centered))


def cosine_similarity(vec1: NDArray, vec2: NDArray) -> float:
    alpha = np.dot(vec1.T, vec2)
    dist_1 = np.linalg.norm(vec1)
    dist_2 = np.linalg.norm(vec2)
    return alpha / (dist_1 * dist_2)


vec_1 = np.array([0, 1, 2, 3])
vec_2 = vec_1 + 100

print(correlation(vec_1, vec_2))
print(cosine_similarity(vec_1, vec_2))

ex2_results = np.zeros((101, 3))
vec3 = np.arange(4)
for i in range(-50, 51):
    vec4 = vec3 + i
    ex2_results[i,0] = i
    ex2_results[i,1] = correlation(vec3, vec4)
    ex2_results[i,2] = cosine_similarity(vec3, vec4)

plt.scatter(ex2_results[:,0], ex2_results[:,2], label="cosine")
plt.scatter(ex2_results[:,0], ex2_results[:,1], label="correlation")
plt.legend()
plt.show()

from scipy.stats import pearsonr