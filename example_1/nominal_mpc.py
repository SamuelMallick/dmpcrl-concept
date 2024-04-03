import logging
import pickle

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env import LtiSystem
from gymnasium.wrappers import TimeLimit
from model import get_centralized_dynamics, get_real_model
from mpcrl.agents.agent import Agent
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

np.random.seed(0)

SAVE = False
TRUE_MODEL = True

n = 3
nx_l = 2
nu_l = 1
x_bnd, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
w = LtiSystem.w


class NominalMPC(Mpc[cs.SX]):
    """A simple randomised scenario based MPC."""

    horizon = 10
    discount_factor = 1

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # action is normal for scenario
        u, _ = self.action("u", n * nu_l, lb=-1, ub=1)

        # state needs to be done manually as we have one state per scenario
        x, _ = self.state("x", n * nx_l)

        s, _, _ = self.variable(
            f"s",
            (n * nx_l, N),
            lb=0,
        )  # n in first dim as only cnstr on theta

        # dynamics
        if TRUE_MODEL:
            A_l, B_l, A_c_l = get_real_model()
        else:
            A_l = np.asarray([[1, 0.25], [0, 1]])
            B_l = np.asarray([[0.0312], [0.25]])
            A_c_l = np.array([[0, 0], [0, 0]])

        A, B = get_centralized_dynamics(n, nx_l, A_l, B_l, A_c_l)
        self.set_dynamics(lambda x, u: A @ x + B @ u, n_in=2, n_out=1)

        # state constraints
        self.constraint(f"x_lb", x_bnd[0] - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", x_bnd[1] + s)

        # create cost matrices combining all scenarios
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            +0.5
            * cs.sum2(
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w @ s)
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
mpc = NominalMPC()
agent = Log(
    Agent(mpc, fixed_parameters={}),
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
        f"nom_true_model_{TRUE_MODEL}" + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
