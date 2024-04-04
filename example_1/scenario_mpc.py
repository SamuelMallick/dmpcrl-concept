import logging
import pickle

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env import LtiSystem
from gymnasium.wrappers import TimeLimit
from model import (
    get_centralized_dynamics,
    get_bounds,
    get_model_details,
    get_true_model,
    get_model_sample,
)
from mpcrl.agents.agent import Agent
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

np.random.seed(0)

SAVE = True
TRUE_MODEL = True

n, nx_l, nu_l = get_model_details()
num_scenarios = 100
x_bnd_l, u_bnd_l, _ = get_bounds()
# create bounds for global state and controls
x_bnd = np.tile(x_bnd_l, n)
u_bnd = np.tile(u_bnd_l, n)
w = np.tile(
    [[1.2e2, 1.2e2]], (1, n)
)  # penalty weight for constraint violations in cost


class ScenarioMpc(Mpc[cs.SX]):
    """A simple randomised scenario based MPC."""

    horizon = 10
    discount_factor = 1

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        disturb = self.parameter("disturb", (n * nx_l * num_scenarios, 1))

        self.fixed_parameter_dict = {
            "disturb": np.random.uniform(-1e-1, 0, (n * nx_l * num_scenarios, 1)),
        }

        # action is normal for scenario
        u, _ = self.action("u", n * nu_l, lb=-1, ub=1)

        # state needs to be done manually as we have one state per scenario
        x = self.nlp.variable(
            "x",
            (n * nx_l * num_scenarios, self._prediction_horizon + 1),
            -np.inf,
            np.inf,
        )[0]
        x0 = self.nlp.parameter("x_0", (n * nx_l, 1))
        self.nlp.constraint("x_0", x[:, 0], "==", cs.repmat(x0, num_scenarios, 1))
        self._states["x"] = x
        self._initial_states["x_0"] = x0

        s, _, _ = self.variable(
            f"s",
            (n * nx_l * num_scenarios, N),
            lb=0,
        )  # n in first dim as only cnstr on theta

        # dynamics
        if TRUE_MODEL:
            A_l, B_l, A_c_l = get_true_model()
            A, B = get_centralized_dynamics(A_l, B_l, A_c_l)

            # create dynamics matrices combining all scenarios
            A_full = block_diag(*[A] * num_scenarios)
            B_full = np.tile(B, (num_scenarios, 1))

            self.set_dynamics(
                lambda x, u: A_full @ x + B_full @ u + disturb, n_in=2, n_out=1
            )
        else:
            A = []
            B = []
            for i in range(num_scenarios):
                A_l, B_l, A_c_l = get_model_sample()
                A_temp, B_temp = get_centralized_dynamics(A_l, B_l, A_c_l)
                A.append(A_temp)
                B.append(B_temp)

            A_full = block_diag(*A)
            B_full = np.vstack(B)

            self.set_dynamics(
                lambda x, u: A_full @ x + B_full @ u + disturb, n_in=2, n_out=1
            )

        # state constraints
        x_bnd_full = np.tile(x_bnd, num_scenarios)
        self.constraint(f"x_lb", x_bnd_full[0] - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", x_bnd_full[1] + s)

        # create cost matrices combining all scenarios
        w_full = np.tile(w, num_scenarios)
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            +0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w_full @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 1000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


env = LtiSystem()
num_eps = 1
ep_len = 100
env = MonitorEpisodes(TimeLimit(env, max_episode_steps=int(ep_len)))
mpc = ScenarioMpc()
agent = Log(
    Agent(mpc, fixed_parameters=mpc.fixed_parameter_dict),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)
returns = agent.evaluate(env=env, episodes=num_eps, seed=1)

print(f"Total returns = {sum(returns)}")

# extract data
if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

if SAVE:
    with open(
        f"scen_{num_scenarios}_truemod_{TRUE_MODEL}" + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
