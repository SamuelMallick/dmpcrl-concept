from typing import Any

import casadi as cs
import gymnasium as gym

# import networkx as netx
import numpy as np
import numpy.typing as npt
from model import (
    get_bounds,
    get_centralized_dynamics,
    get_model_details,
    get_true_model,
)


class LtiSystem(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A discrete time network of LTI systems."""

    n, nx_l, nu_l = get_model_details()
    A_l, B_l, A_c_l = get_true_model()
    A, B = get_centralized_dynamics(A_l, B_l, A_c_l)
    nx = n * nx_l  # number of states
    nu = n * nu_l  # number of inputs

    x_bnd_l, u_bnd_l, noise_bnd = get_bounds()
    # create bounds for global state and controls
    x_bnd = np.tile(x_bnd_l, n)
    u_bnd = np.tile(u_bnd_l, n)

    w = np.tile([[1.2e2, 1.2e2]], (1, n))  # penalty weight for bound violations

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the network. An x0 can be passed in the options dict."""
        super().reset(seed=seed, options=options)
        if options is not None and "x0" in options:
            self.x = options["x0"]
        else:
            self.x = np.tile([0, 0.15], self.n).reshape(self.nx, 1)
        return self.x, {}

    def get_stage_cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Computes the stage cost `L(s,a)`."""
        lb, ub = self.x_bnd
        return 0.5 * float(
            np.square(state).sum()
            + 0.5 * np.square(action).sum()
            + self.w @ np.maximum(0, lb[:, np.newaxis] - state)
            + self.w @ np.maximum(0, state - ub[:, np.newaxis])
        )

    def get_dist_stage_cost(self, state: np.ndarray, action: np.ndarray) -> list[float]:
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
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Steps the network."""
        action = action.full()  # convert action from casadi DM to numpy array
        x_new = self.A @ self.x + self.B @ action
        noise = self.np_random.uniform(*self.noise_bnd).reshape(-1, 1)
        # apply noise only to first state dimension of each agent
        x_new[np.arange(0, self.nx, self.nx_l)] += noise

        r = self.get_stage_cost(self.x, action)
        r_dist = self.get_dist_stage_cost(self.x, action)
        self.x = x_new

        return x_new, r, False, False, {"r_dist": r_dist}
