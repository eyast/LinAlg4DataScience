import numpy as np
from numpy.typing import NDArray

def matrix_norm(matrix: NDArray) -> float:
    r"""
    Calculates the Frobenius norm.
    """
    square = matrix @ matrix.T
    trace = np.trace(square)
    norm = np.sqrt(trace)
    return norm

def v_projection(b: NDArray, a: NDArray)-> NDArray:
    r"""Projects vector b on vector a.
    Vector a is the reference"""
    assert a.shape == b.shape
    beta =  np.dot(b, a) / np.dot(a, a)
    return beta * a

def is_orthogonal(vec1, vec2):
    r"""Asserts that two vectors are orthogonal"""
    return np.allclose(np.dot(vec1, vec2), 0)

def is_matrix_orthogonal (matrix:NDArray) -> bool:
    return np.allclose(
                        matrix.T @ matrix,
                        np.eye(
                           matrix.shape[1]
                        )
            )

def gs(mat: NDArray) -> tuple[NDArray,NDArray]:
    r"""
    Decomposes a matrix into Q and R.
    """
    Q = np.zeros_like(mat, dtype=float)
    Q[:,0] = mat[:,0].copy() / np.linalg.norm(mat[:,0])
    cols = mat.shape[1]
    for col in range(1, cols):
        q = mat[:,col].copy()
        for k in range(col):
            print(f"Calculating column q_{col} with projection x_{k}")
            # proj = v_projection(mat[:,col], Q[:,k])
            proj = v_projection(q, Q[:,k])
            q = q - proj
        Q[:,col] = q / np.linalg.norm(q)
    R = Q.T @ mat
    return Q, R