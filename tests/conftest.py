import pytest
import numpy as np


@pytest.fixture
def matrices(name: str) -> np.ndarray:
    A = {"identity": np.diag([5., 5., 5., 5.]),
         "forsythe": np.array([[0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1],
                               [1.49012e-8, 0, 0, 0]]),
         "gravity": np.array([[4.0, 1.41421,  0.357771, 0.126491],
                              [1.41421, 4.0, 1.41421, 0.357771],
                              [0.357771, 1.41421, 4.0, 1.41421],
                              [0.126491, 0.357771, 1.41421, 4.0]]),
         "fiedler": np.array([[0, 1, 2, 3],
                              [1, 0, 1, 2],
                              [2, 1, 0, 1],
                              [3, 2, 1, 0]]),
         "hilbert": np.array([[1., 1/2, 1/3, 1/4],
                              [1/2, 1/3, 1/4, 1/5],
                              [1/3, 1/4, 1/5, 1/6],
                              [1/4, 1/5, 1/6, 1/7]])
         }

    return A[name]
