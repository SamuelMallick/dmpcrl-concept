import casadi as cs
import numpy as np
from scipy.linalg import block_diag

np.random.seed(0)

n = 3  # number of agents
nx_l = 2  # local state dimension
nu_l = 1  # local control dimension


def get_model_details():
    """Returns number of agents (n), local state dim (nx_l), and local control dim (nu_l)."""
    return n, nx_l, nu_l


# bounds
x_bnd_l = np.array([[0, -1], [1, 1]])  # local state bounds x_bnd[0] <= x <= x_bnd[1]
u_bnd_l = np.array([[-1], [1]])  # local control bounds u_bnd[0] <= u <= u_bnd[1]
noise_bnd = np.array([[-1e-1], [0]])  # uniform noise bounds for process noise


def get_bounds():
    """Returns the bounds on states (x_bnd_l), controls (u_bnd_l), and process noise (noise_bnd)."""
    return x_bnd_l, u_bnd_l, noise_bnd


# true unknown model, '_l' subscript indicates that this is a local system matrix rather than centralized
# A_c is the coupling matrix and is assumed to be to same for all couplings i.e., A_12 = A_21 = A_32 etc.
A_l_true = np.array([[0.9, 0.35], [0, 1.1]])
B_l_true = np.array([[0.0813], [0.2]])
A_c_l_true = np.array([[0, 0], [0, -0.1]])

# local innacurate model
A_l_inac = np.asarray([[1, 0.25], [0, 1]])
B_l_inac = np.asarray([[0.0312], [0.25]])
A_c_l_inac = np.array([[0, 0], [0, 0]])

# adjacency matrix of coupling in network
Adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)


def get_adj():
    """Returns the adjacany matrix for the network Adj. Adj[i,j] = 1 indicates i,j are coupled."""
    return Adj


def get_true_model():
    """Returns the true model for a local system."""
    return A_l_true, B_l_true, A_c_l_true


def get_inac_model():
    """Returns an inaccurate model for a local system."""
    return A_l_inac, B_l_inac, A_c_l_inac


# distribution of model uncertainty. A uniform distribution centrered around the known inaccurate model
range_A = 0.15
range_B = 0.1


def get_model_sample():
    """Returns a sample of an innacurate model for a local system."""
    A_l_sample = A_l_inac + np.array(
        [
            [
                np.random.uniform(-range_A, range_A),
                np.random.uniform(-range_A, range_A),
            ],
            [0, np.random.uniform(-range_A, range_A)],
        ]
    )
    A_c_l_sample = A_c_l_inac + np.array(
        [[0, 0], [0, np.random.uniform(-range_A, range_A)]]
    )
    B_l_sample = B_l_inac + np.array(
        [[np.random.uniform(-range_B, range_B)], [np.random.uniform(-range_B, range_B)]]
    )
    return A_l_sample, B_l_sample, A_c_l_sample


def get_centralized_dynamics(
    A_l: np.ndarray,
    B_l: np.ndarray,
    A_c: np.ndarray,
):
    """Returns a centralized representation of the dynamics from the local dynamics."""
    A = np.zeros((n * nx_l, n * nx_l))  # global state-space matrix A
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_l
            elif Adj[i, j] == 1:
                A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c
                A[nx_l * j : nx_l * (j + 1), nx_l * i : nx_l * (i + 1)] = A_c
    B = block_diag(*[B_l] * n)  # global state-space matix B
    return A, B


def get_learnable_centralized_dynamics(
    A_list: list[cs.SX],
    B_list: list[cs.SX],
    A_c_list: list[list[cs.SX]],
):
    """Returns a centralized representation of dynamics from symbolic local dynamics matrices."""
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
    return A, B
