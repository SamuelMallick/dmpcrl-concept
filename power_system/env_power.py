from typing import Any, Dict, Optional, Tuple

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from model_Hycon2 import get_cent_model, get_model_details, get_P_tie
from scipy.linalg import block_diag

np.random.seed(1)

(
    n,
    nx_l,
    nu_l,
    Adj,
    ts,
    prediction_length,
    discount_factor,
    u_lim,
    theta_lim,
    w,
    load_noise_bnd,
) = get_model_details()  # see model_Hycon2 file for definition of each


class PowerSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Cont time network of four power system areas connected with tie lines."""

    A, B, L = get_cent_model(discrete=False)  # Get continuous centralised model

    # stage cost params
    Q_x_l = np.diag((500, 0.01, 0.01, 10))
    Q_x = block_diag(*([Q_x_l] * n))
    Q_u_l = 10
    Q_u = block_diag(*([Q_u_l] * n))

    load = np.array([[0], [0], [0], [0]])  # load ref points - changed in step function
    x_o = np.zeros(
        (n * nx_l, 1)
    )  # set points for state and action - set with load changes
    u_o = np.zeros((n * nu_l, 1))

    phi_weight = 0.5  # weight given to power transfer term in stage cost
    P_tie_list = get_P_tie()  # true power transfer coefficients

    step_counter = 1

    def __init__(self) -> None:
        super().__init__()

        # set-up continuous time integrator for dynamics simulation
        x = cs.SX.sym("x", self.A.shape[1])
        u = cs.SX.sym("u", self.B.shape[1])
        l = cs.SX.sym("l", self.L.shape[1])
        p = cs.vertcat(u, l)
        x_new = self.A @ x + self.B @ u + self.L @ l
        ode = {"x": x, "p": p, "ode": x_new}
        self.integrator = cs.integrator(
            "env_integrator",
            "cvodes",
            ode,
            0,
            ts,
            {"abstol": 1e-8, "reltol": 1e-8},
        )

    def set_points(self, load_val):
        """Calculate state and action nset points based on load value."""
        x_o_val = np.array(
            [
                [
                    0,
                    0,
                    load_val[0, :].item(),
                    load_val[0, :].item(),
                    0,
                    0,
                    load_val[1, :].item(),
                    load_val[1, :].item(),
                    0,
                    0,
                    load_val[2, :].item(),
                    load_val[2, :].item(),
                    0,
                    0,
                    load_val[3, :].item(),
                    load_val[3, :].item(),
                ]
            ]
        ).T
        u_o_val = load_val.copy()
        return x_o_val, u_o_val

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.zeros((n * nx_l, 1))
        self.load = np.zeros((n, 1))
        self.x_o, self.u_o = self.set_points(self.load)
        self.step_counter = 1
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes stage cost L(s,a)"""
        return (
            (state - self.x_o).T @ self.Q_x @ (state - self.x_o)
            + (action - self.u_o).T @ self.Q_u @ (action - self.u_o)
            + self.phi_weight
            * ts
            * (  # power transfer term
                sum(
                    np.abs(self.P_tie_list[i, j] * (state[i * nx_l] - state[j * nx_l]))
                    for j in range(n)
                    for i in range(n)
                    if Adj[i, j] == 1
                )
            )
            # pulling out thetas via slice [0, 4, 8, 12]
            + w.T @ np.maximum(0, -np.ones((n, 1)) * theta_lim - state[[0, 4, 8, 12]])
            + w.T @ np.maximum(0, state[[0, 4, 8, 12]] - np.ones((n, 1)) * theta_lim)
        )

    def get_dist_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> list[float]:
        stage_costs = [
            (
                state[nx_l * i : nx_l * (i + 1), :]
                - self.x_o[nx_l * i : nx_l * (i + 1), :]
            ).T
            @ self.Q_x_l
            @ (
                state[nx_l * i : nx_l * (i + 1), :]
                - self.x_o[nx_l * i : nx_l * (i + 1), :]
            )
            + (
                action[nu_l * i : nu_l * (i + 1), :]
                - self.u_o[nu_l * i : nu_l * (i + 1), :]
            ).T
            @ self.Q_u_l
            @ (
                action[nu_l * i : nu_l * (i + 1), :]
                - self.u_o[nu_l * i : nu_l * (i + 1), :]
            )
            + self.phi_weight
            * ts
            * (
                sum(
                    np.abs(self.P_tie_list[i, j] * (state[i * nx_l] - state[j * nx_l]))
                    for j in range(n)
                    if Adj[i, j] == 1
                )
            )
            + w[0] @ np.maximum(0, -theta_lim - state[n * i])
            + w[0] @ np.maximum(0, state[n * i] - theta_lim)
            for i in range(n)
        ]
        return np.asarray(stage_costs).reshape(n, 1)

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the system."""
        r = float(self.get_stage_cost(self.x, action))
        r_dist = self.get_dist_stage_cost(self.x, action)

        # Change load according to scenario
        if self.step_counter == 5:
            self.load = np.array([[0.15, 0, 0, 0]]).T
            self.x_o, self.u_o = self.set_points(self.load)
        elif self.step_counter == 15:
            self.load = np.array([[0.15, -0.15, 0, 0]]).T
            self.x_o, self.u_o = self.set_points(self.load)
        elif self.step_counter == 20:
            self.load = np.array([[0.15, -0.15, 0.12, 0]]).T
            self.x_o, self.u_o = self.set_points(self.load)
        elif self.step_counter == 40:
            self.load = np.array([[0.15, -0.15, -0.12, 0.28]]).T
            self.x_o, self.u_o = self.set_points(self.load)

        load_noise = np.random.uniform(-load_noise_bnd, load_noise_bnd, (n, 1))
        l = self.load + load_noise
        x_new = self.integrator(x0=self.x, p=cs.vertcat(action, l))["xf"]
        self.x = x_new
        self.step_counter += 1
        return x_new, r, False, False, {"r_dist": r_dist}
