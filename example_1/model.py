import contextlib

import casadi as cs

# import networkx as netx
import numpy as np
import numpy.typing as npt

Adj = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32
)  # adjacency matrix of coupling in network


def get_adj():
    return Adj


def get_centralized_dynamics(
    n: int,
    nx_l: int,
    A_l,
    B_l,
    A_c: npt.NDArray[np.floating],
):
    """Creates the centralized representation of the dynamics from the real dynamics."""
    A = cs.SX.zeros(n * nx_l, n * nx_l)  # global state-space matrix A
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_l
            elif Adj[i, j] == 1:
                A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c
                A[nx_l * j : nx_l * (j + 1), nx_l * i : nx_l * (i + 1)] = A_c
    with contextlib.suppress(RuntimeError):
        A = cs.evalf(A).full()
    B = cs.diagcat(*(B_l for _ in range(n)))  # global state-space matix B
    with contextlib.suppress(RuntimeError):
        B = cs.evalf(B).full()
    return A, B


def get_learnable_centralized_dynamics(
    n: int,
    nx_l: int,
    nu_l: int,
    A_list: list,
    B_list: list,
    A_c_list: list[list[npt.NDArray[np.floating]]],
    B_c_list: list[list[npt.NDArray[np.floating]]],
):
    """Creates the centralized representation of the dynamics from the learnable dynamics."""
    A = cs.SX.zeros(n * nx_l, n * nx_l)  # global state-space matrix A
    B = cs.SX.zeros(n * nx_l, n * nu_l)  # global state-space matix B
    for i in range(n):
        for j in range(n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_list[i]
                B[nx_l * i : nx_l * (i + 1), nu_l * i : nu_l * (i + 1)] = B_list[i]
            else:
                if Adj[i, j] == 1:
                    A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c_list[
                        i
                    ][j]
    with contextlib.suppress(RuntimeError):
        A = cs.evalf(A).full()
    with contextlib.suppress(RuntimeError):
        B = cs.evalf(B).full()
    return A, B
