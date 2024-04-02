from typing import Any

import casadi as cs
import gymnasium as gym

# import networkx as netx
import numpy as np
import numpy.typing as npt


class LtiSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A discrete time network of LTI systems."""

    n = 3  # number of agents
    nx_l = 2  # number of agent states
    nu_l = 1  # number of agent inputs

    A_l = np.array([[0.9, 0.35], [0, 1.1]])  # agent state-space matrix A
    B_l = np.array([[0.0813], [0.2]])  # agent state-space matrix B
    A_c = np.array([[0, 0], [0, -0.1]])  # common coupling state-space matrix
    A, B = get_centralized_dynamics(n, nx_l, A_l, B_l, A_c)
    nx = n * nx_l  # number of states
    nu = n * nu_l  # number of inputs

    w = np.tile([[1.2e2, 1.2e2]], (1, n))  # agent penalty weight for bound violations
    x_bnd = np.tile([[0, -1], [1, 1]], (1, n))
    a_bnd = np.tile([[-1], [1]], (1, n))
    e_bnd = np.tile([[-1e-1], [0]], (1, n))  # uniform noise bounds

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.tile([0, 0.15], self.n).reshape(
            self.nx, 1
        )  # + np.random.rand(6, 1)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""
        lb, ub = self.x_bnd
        return 0.5 * float(
            np.square(state).sum()
            + 0.5 * np.square(action).sum()
            + self.w @ np.maximum(0, lb[:, np.newaxis] - state)
            + self.w @ np.maximum(0, state - ub[:, np.newaxis])
        )

    def get_dist_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> list[float]:
        """Computes the stage cost for each agent `L(s_i,a_i)`."""
        lb, ub = self.x_bnd
        stage_costs = [
            0.5
            * float(
                np.square(state[self.nx_l * i : self.nx_l * (i + 1), :]).sum()
                + 0.5 * np.square(action[self.nu_l * i : self.nu_l * (i + 1), :]).sum()
                + self.w[:, self.nx_l * i : self.nx_l * (i + 1)]
                @ np.maximum(
                    0,
                    lb[self.nx_l * i : self.nx_l * (i + 1), np.newaxis]
                    - state[self.nx_l * i : self.nx_l * (i + 1), :],
                )
                + self.w[:, self.nx_l * i : self.nx_l * (i + 1)]
                @ np.maximum(
                    0,
                    state[self.nx_l * i : self.nx_l * (i + 1), :]
                    - ub[self.nx_l * i : self.nx_l * (i + 1), np.newaxis],
                )
            )
            for i in range(self.n)
        ]
        return stage_costs

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the LTI system."""
        action = action.full()
        x_new = self.A @ self.x + self.B @ action

        noise = self.np_random.uniform(*self.e_bnd).reshape(-1, 1)
        x_new[np.arange(0, self.nx, self.nx_l)] += noise

        r = self.get_stage_cost(self.x, action)
        r_dist = self.get_dist_stage_cost(self.x, action)
        self.x = x_new

        return x_new, r, False, False, {"r_dist": r_dist}
