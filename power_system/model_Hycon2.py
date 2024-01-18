import pickle
from typing import Dict, Tuple

import casadi as cs

# import networkx as netx
import numpy as np

from dmpcrl.utils.discretisation import zero_order_hold

np.random.seed(1)

# Model from
# Hycon2 benchmark paper 2012 S. Riverso, G. Ferrari-Tracate

# real parameters of the power system - each is a list containing values for each of the four areas

n = 4  # num agents
nx_l = 4  # agent state dim
nu_l = 1  # agent control dim

N = 5  # MPC prediction horizon
discount_factor = 0.9  # discount factor in MPC cost

u_lim = np.array([[0.2], [0.1], [0.3], [0.1]])  # limit on agent control actions
theta_lim = 0.1  # limit on first state of each agent
w = 500 * np.ones((n, 1))  # penalty on state viols
load_noise_bnd = 1e-1  # uniform noise bound on load noise

Adj = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

H_list = [12.0, 10, 8, 8]
R_list = [0.05, 0.0625, 0.08, 0.08]
D_list = [0.7, 0.9, 0.9, 0.7]
T_t_list = [0.65, 0.4, 0.3, 0.6]
T_g_list = [0.1, 0.1, 0.1, 0.1]
P_tie = np.array(
    [
        [0, 4.0, 0, 0],
        [4.0, 0, 2.0, 0],
        [0, 2.0, 0, 2.0],
        [0, 0, 2.0, 0],
    ]
)  # entri (i,j) represent P val between areas i and j
ts = 1  # time-step for discretisation


def get_P_tie():
    return P_tie


def get_model_details() -> (
    Tuple[
        int,
        int,
        int,
        np.ndarray,
        float,
        int,
        float,
        np.ndarray,
        float,
        np.ndarray,
        float,
    ]
):
    return (
        n,
        nx_l,
        nu_l,
        Adj,
        ts,
        N,
        discount_factor,
        u_lim,
        theta_lim,
        w,
        load_noise_bnd,
    )


def get_cent_model(discrete: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get A, B and L matrices for the centralised system Ax + Bu + Ld. If discrete the continuous dynamics are discretised using ZOH."""
    A_l = [
        np.array(
            [
                [0, 1, 0, 0],
                [
                    -sum(P_tie[i, :n]) / (2 * H_list[i]),
                    -D_list[i] / (2 * H_list[i]),
                    1 / (2 * H_list[i]),
                    0,
                ],
                [0, 0, -1 / T_t_list[i], 1 / T_t_list[i]],
                [0, -1 / (R_list[i] * T_g_list[i]), 0, -1 / T_g_list[i]],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [0], [1 / T_g_list[i]]]) for i in range(n)]
    L_l = [np.array([[0], [-1 / (2 * H_list[i])], [0], [0]]) for i in range(n)]

    # coupling

    A_c = [
        [
            np.array(
                [
                    [0, 0, 0, 0],
                    [P_tie[i, j] / (2 * H_list[i]), 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    # global
    A = np.vstack(
        (
            np.hstack((A_l[0], A_c[0][1], A_c[0][2], A_c[0][3])),
            np.hstack((A_c[1][0], A_l[1], A_c[1][2], A_c[1][3])),
            np.hstack((A_c[2][0], A_c[2][1], A_l[2], A_c[2][3])),
            np.hstack((A_c[3][0], A_c[3][1], A_c[3][2], A_l[3])),
        )
    )
    B = np.vstack(
        (
            np.hstack((B_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), B_l[1], np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), B_l[2], np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), B_l[3])),
        )
    )
    L = np.vstack(
        (
            np.hstack((L_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), L_l[1], np.zeros((n, 1)), np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), L_l[2], np.zeros((n, 1)))),
            np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), L_l[3])),
        )
    )
    if not discrete:
        return A, B, L
    else:
        B_comb = np.hstack((B, L))
        A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
        B_d = B_d_comb[:, :n]
        L_d = B_d_comb[:, n:]
        return A_d, B_d, L_d


# initial guesses for each learnable parameter for each agent - except for P_tie
pars_init = [
    {
        "H": (H_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "R": (R_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "D": (D_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "T_t": (T_t_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "T_g": (T_g_list[i] + np.random.normal(0, 0)) * np.ones((1,)),
        "theta_lb": 0 * np.ones((1,)),
        "theta_ub": 0 * np.ones((1,)),
        "V0": 0 * np.ones((1,)),
        "b": 0 * np.ones((nx_l,)),
        "f_x": 0 * np.ones((nx_l, 1)),
        "f_u": 0 * np.ones((nu_l, 1)),
        "Q_x": np.diag((500, 0.1, 0.1, 10)),
        "Q_u": 10 * np.ones((1,)),
    }
    for i in range(n)
]


def get_pars_init_list() -> list[Dict]:
    """Get initial guesses for learnable parameters (exluding P_tie)."""
    return pars_init


# create initial guesses as a purturbation of initial P_tie vals
norm_lim = 2.0
P_tie_init = P_tie.copy()
for i in range(n):
    for j in range(n):
        if P_tie_init[i, j] != 0:
            P_tie_init[i, j] += np.random.uniform(-norm_lim, norm_lim)


def get_P_tie_init() -> np.ndarray:
    """Get initial guesses for learnable P_tie values."""
    return P_tie_init


def get_learnable_dynamics(
    H_list: list[cs.SX],
    R_list: list[cs.SX],
    D_list: list[cs.SX],
    T_t_list: list[cs.SX],
    T_g_list: list[cs.SX],
    P_tie_list_list: list[list[cs.SX]],
):
    """Get symbolic A, B and L matrices for the centralised system Ax + Bu + Ld. Always discretised."""
    A_l = [
        cs.blockcat(
            [
                [0, 1, 0, 0],
                [
                    -sum(P_tie_list_list[i]) / (2 * H_list[i]),
                    -D_list[i] / (2 * H_list[i]),
                    1 / (2 * H_list[i]),
                    0,
                ],
                [0, 0, -1 / T_t_list[i], 1 / T_t_list[i]],
                [0, -1 / (R_list[i] * T_g_list[i]), 0, -1 / T_g_list[i]],
            ]
        )
        for i in range(n)
    ]
    B_l = [cs.blockcat([[0], [0], [0], [1 / T_g_list[i]]]) for i in range(n)]
    L_l = [cs.blockcat([[0], [-1 / (2 * H_list[i])], [0], [0]]) for i in range(n)]

    # coupling

    A_c = [
        [
            cs.blockcat(
                [
                    [0, 0, 0, 0],
                    [P_tie_list_list[i][j] / (2 * H_list[i]), 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    # global
    A = cs.vertcat(
        cs.horzcat(A_l[0], A_c[0][1], A_c[0][2], A_c[0][3]),
        cs.horzcat(A_c[1][0], A_l[1], A_c[1][2], A_c[1][3]),
        cs.horzcat(A_c[2][0], A_c[2][1], A_l[2], A_c[2][3]),
        cs.horzcat(A_c[3][0], A_c[3][1], A_c[3][2], A_l[3]),
    )
    B = cs.vertcat(
        cs.horzcat(B_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), B_l[1], np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), B_l[2], np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), B_l[3]),
    )
    L = cs.vertcat(
        cs.horzcat(L_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), L_l[1], np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), L_l[2], np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), L_l[3]),
    )

    # discretise
    B_comb = cs.horzcat(B, L)
    # A_d, B_d_comb = forward_euler(A, B_comb, ts)
    A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
    B_d = B_d_comb[:, :n]
    L_d = B_d_comb[:, n:]
    return A_d, B_d, L_d


def get_learnable_dynamics_local(H, R, D, T_t, T_g, P_tie_list):
    """Get symbolic matrices A_i, B_i, L_i and A_ij for an agent. Always discretised."""
    A = cs.blockcat(
        [
            [0, 1, 0, 0],
            [
                -sum(P_tie_list) / (2 * H),
                -D / (2 * H),
                1 / (2 * H),
                0,
            ],
            [0, 0, -1 / T_t, 1 / T_t],
            [0, -1 / (R * T_g), 0, -1 / T_g],
        ]
    )
    B = cs.blockcat([[0], [0], [0], [1 / T_g]])
    L = cs.blockcat([[0], [-1 / (2 * H)], [0], [0]])

    A_c_list = []
    for i in range(len(P_tie_list)):
        A_c_list.append(
            cs.blockcat(
                [
                    [0, 0, 0, 0],
                    [P_tie_list[i] / (2 * H), 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        )

    B_comb = cs.horzcat(B, L, *A_c_list)
    A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
    B_d = B_d_comb[:, :nu_l]
    L_d = B_d_comb[:, nu_l : 2 * nu_l]
    A_d_c_list = []
    for i in range(len(P_tie_list)):
        A_d_c_list.append(B_d_comb[:, 2 * nu_l + i * nx_l : 2 * nu_l + (i + 1) * nx_l])
    return A_d, B_d, L_d, A_d_c_list


learned_file = "data/power_data/line_40/distributed_con.pkl"


def get_learned_pars_init_list(centralised=False):
    with open(
        learned_file,
        "rb",
    ) as file:
        pickle.load(file)
        pickle.load(file)
        pickle.load(file)
        pickle.load(file)
        param_list = pickle.load(file)

    learned_pars_init = []
    for i in range(n):
        learned_pars_init.append(
            {
                "H": H_list[i],
                "R": R_list[i],
                "D": D_list[i],
                "T_t": T_t_list[i],
                "T_g": T_g_list[i],
                "theta_lb": param_list[f"theta_lb_{i}"][-1],
                "theta_ub": param_list[f"theta_ub_{i}"][-1],
                "V0": param_list[f"V0_{i}"][-1],
                "b": param_list[f"b_{i}"][-1],
                "f_x": param_list[f"f_x_{i}"][-1, :],
                "f_u": param_list[f"f_u_{i}"][-1, :],
                "Q_x": param_list[f"Q_x_{i}"][-1, :],
                "Q_u": param_list[f"Q_u_{i}"][-1, :],
            }
        )
    return learned_pars_init


def get_learned_P_tie_init(centralised=False):
    with open(
        learned_file,
        "rb",
    ) as file:
        pickle.load(file)
        pickle.load(file)
        pickle.load(file)
        pickle.load(file)
        param_list = pickle.load(file)
    learned_P_tie = np.zeros((n, n))  # TODO: make this work also for distributed data
    if centralised:
        for i in range(n):
            for j in range(n):
                if f"P_tie_{i}_{j}" in param_list:
                    learned_P_tie[i, j] = param_list[f"P_tie_{i}_{j}"][-1]
    else:
        for i in range(n):
            if f"P_tie_0_{i}" in param_list:
                learned_P_tie[i, i - 1] = param_list[f"P_tie_0_{i}"][-1]
            if f"P_tie_1_{i}" in param_list:
                learned_P_tie[i, i + 1] = param_list[f"P_tie_1_{i}"][-1]
    return learned_P_tie
