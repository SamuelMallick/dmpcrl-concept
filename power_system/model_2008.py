from typing import Dict, Tuple

import casadi as cs

# import networkx as netx
import numpy as np

from dmpcrl.utils.discretisation import forward_euler

# real parameters of the power system - each is a list containing value for each of the four areas

# From Venkat, Aswin N., et al.
# "Distributed MPC strategies with application to power system automatic generation
# control." IEEE transactions on control systems technology 16.6 (2008): 1192-1206.

n = 4
nx_l = 4
nu_l = 1

D_list = [3.0, 0.275, 2.0, 2.75]
R_f_list = [0.03, 0.07, 0.04, 0.03]
M_a_list = [4, 40, 35, 10]
# M_a_list = [20, 100, 50, 60]
# T_CH_list = [5, 10, 20, 10]
T_CH_list = [50, 100, 200, 100]
# T_G_list = [4, 25, 15, 5]
T_G_list = [40, 250, 150, 50]
T_tie_list = [0, 2.54, 1.5, 2.5]
ts = 1  # time-step

# construct real dynamics - subscript l is for local components


def dynamics_from_parameters(D, R_f, M_a, T_CH, T_G, T_tie, ts):
    A_l = [
        np.array(
            [
                [-D[i] / M_a[i], 1 / M_a[i], 0, 1 / M_a[i]],
                [0, -1 / T_CH[i], 1 / T_CH[i], 0],
                [-1 / (R_f[i] * T_G[i]), 0, -1 / T_G[i], 0],
                [-T_tie[i], 0, 0, 0],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [1 / T_G[i]], [0]]) for i in range(n)]
    A_l_load = [np.array([[-1 / M_a[i]], [0], [0], [0]]) for i in range(n)]

    # build each element of global. including coupling - this coupling is not symetrical so it is done by hand

    # agent 0
    A_00 = A_l[0]
    A_01 = np.zeros((nx_l, nx_l))
    A_01[0, 3] = -1.0 / M_a[0]
    A_02 = np.zeros((nx_l, nx_l))
    A_03 = np.zeros((nx_l, nx_l))

    # agent 1
    A_10 = np.zeros((nx_l, nx_l))
    A_10[3, 0] = T_tie[1]
    A_11 = A_l[1]
    A_12 = np.zeros((nx_l, nx_l))
    A_12[0, 3] = -1 / M_a[1]
    A_13 = np.zeros((nx_l, nx_l))

    # agent 2
    A_20 = np.zeros((nx_l, nx_l))
    A_21 = np.zeros((nx_l, nx_l))
    A_21[3, 0] = T_tie[2]
    A_22 = A_l[2]
    A_23 = np.zeros((nx_l, nx_l))
    A_23[0, 3] = -1 / M_a[2]

    # agent 3
    A_30 = np.zeros((nx_l, nx_l))
    A_31 = np.zeros((nx_l, nx_l))
    A_32 = np.zeros((nx_l, nx_l))
    A_32[3, 0] = T_tie[3]
    A_33 = A_l[3]

    # global
    A = np.vstack(
        (
            np.hstack((A_00, A_01, A_02, A_03)),
            np.hstack((A_10, A_11, A_12, A_13)),
            np.hstack((A_20, A_21, A_22, A_23)),
            np.hstack((A_30, A_31, A_32, A_33)),
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
    A_load = np.vstack(
        (
            np.hstack(
                (A_l_load[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)))
            ),
            np.hstack(
                (np.zeros((n, 1)), A_l_load[1], np.zeros((n, 1)), np.zeros((n, 1)))
            ),
            np.hstack(
                (np.zeros((n, 1)), np.zeros((n, 1)), A_l_load[2], np.zeros((n, 1)))
            ),
            np.hstack(
                (np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), A_l_load[3])
            ),
        )
    )

    # disrctised
    A_d, B_d = forward_euler(A, B, ts)
    A_load_d = ts * A_load

    return A_d, B_d, A_load_d


def learnable_dynamics_from_parameters(D, R_f, M_a, T_CH, T_G, T_tie, ts):
    A_l = [
        np.array(
            [
                [-D[i] / M_a[i], 1 / M_a[i], 0, -1 / M_a[i]],
                [0, -1 / T_CH[i], 1 / T_CH[i], 0],
                [-1 / (R_f[i] * T_G[i]), 0, -1 / T_G[i], 0],
                [-T_tie[i], 0, 0, 0],
            ]
        )
        for i in range(n)
    ]
    B_l = [np.array([[0], [0], [1 / T_G[i]], [0]]) for i in range(n)]
    A_l_load = [np.array([[-1 / M_a[i]], [0], [0], [0]]) for i in range(n)]

    # build each element of global. including coupling - this coupling is not symetrical so it is done by hand

    # agent 0
    A_00 = A_l[0]
    A_01 = cs.SX.zeros(nx_l, nx_l)
    A_01[0, 3] = -1.0 / M_a[0]
    A_02 = cs.SX.zeros(nx_l, nx_l)
    A_03 = cs.SX.zeros(nx_l, nx_l)

    # agent 1
    A_10 = cs.SX.zeros(nx_l, nx_l)
    A_10[3, 0] = T_tie[1]
    A_11 = A_l[1]
    A_12 = cs.SX.zeros(nx_l, nx_l)
    A_12[0, 3] = -1 / M_a[1]
    A_13 = cs.SX.zeros(nx_l, nx_l)

    # agent 2
    A_20 = cs.SX.zeros(nx_l, nx_l)
    A_21 = cs.SX.zeros(nx_l, nx_l)
    A_21[3, 0] = T_tie[2]
    A_22 = A_l[2]
    A_23 = cs.SX.zeros(nx_l, nx_l)
    A_23[0, 3] = -1 / M_a[2]

    # agent 3
    A_30 = cs.SX.zeros(nx_l, nx_l)
    A_31 = cs.SX.zeros(nx_l, nx_l)
    A_32 = cs.SX.zeros(nx_l, nx_l)
    A_32[3, 0] = T_tie[3]
    A_33 = A_l[3]

    # global
    A = cs.vertcat(
        cs.horzcat(A_00, A_01, A_02, A_03),
        cs.horzcat(A_10, A_11, A_12, A_13),
        cs.horzcat(A_20, A_21, A_22, A_23),
        cs.horzcat(A_30, A_31, A_32, A_33),
    )
    B = cs.vertcat(
        cs.horzcat(B_l[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), B_l[1], np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), B_l[2], np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), B_l[3]),
    )
    A_load = cs.vertcat(
        cs.horzcat(A_l_load[0], np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), A_l_load[1], np.zeros((n, 1)), np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), A_l_load[2], np.zeros((n, 1))),
        cs.horzcat(np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)), A_l_load[3]),
    )

    # disrctised
    A_d, B_d = forward_euler(A, B, ts)
    A_load_d = ts * A_load

    return A_d, B_d, A_load_d


def get_cent_model() -> Tuple[np.ndarray, np.ndarray]:
    return dynamics_from_parameters(
        D_list, R_f_list, M_a_list, T_CH_list, T_G_list, T_tie_list, ts
    )


def get_model_dims() -> Tuple[int, int, int]:
    return n, nx_l, nu_l


# initial guesses for each learnable parameter for each agent
learnable_pars_init_0 = {
    "D": 3.0 * np.ones((1,)),
    "R_f": 0.03 * np.ones((1,)),
    "M_a": M_a_list[0] * np.ones((1,)),
    "T_CH": T_CH_list[0] * np.ones((1,)),
    "T_G": T_G_list[0] * np.ones((1,)),
    "T_tie": 0 * np.ones((1,)),
}
learnable_pars_init_1 = {
    "D": 0.275 * np.ones((1,)),
    "R_f": 0.07 * np.ones((1,)),
    "M_a": M_a_list[1] * np.ones((1,)),
    "T_CH": T_CH_list[1] * np.ones((1,)),
    "T_G": T_G_list[1] * np.ones((1,)),
    "T_tie": 2.54 * np.ones((1,)),
}
learnable_pars_init_2 = {
    "D": 2.0 * np.ones((1,)),
    "R_f": 0.04 * np.ones((1,)),
    "M_a": M_a_list[2] * np.ones((1,)),
    "T_CH": T_CH_list[2] * np.ones((1,)),
    "T_G": T_G_list[2] * np.ones((1,)),
    "T_tie": 1.5 * np.ones((1,)),
}
learnable_pars_init_3 = {
    "D": 2.75 * np.ones((1,)),
    "R_f": 0.03 * np.ones((1,)),
    "M_a": M_a_list[3] * np.ones((1,)),
    "T_CH": T_CH_list[3] * np.ones((1,)),
    "T_G": T_G_list[3] * np.ones((1,)),
    "T_tie": 2.5 * np.ones((1,)),
}


def get_learnable_pars_init_list() -> list[Dict]:
    return [
        learnable_pars_init_0,
        learnable_pars_init_1,
        learnable_pars_init_2,
        learnable_pars_init_3,
    ]


def get_learnable_dynamics(D_list, R_f_list, M_a_list, T_CH_list, T_G_list, T_tie_list):
    A, B, A_load = learnable_dynamics_from_parameters(
        D_list, R_f_list, M_a_list, T_CH_list, T_G_list, T_tie_list, ts
    )
    return A, B, A_load
