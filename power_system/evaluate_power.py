import datetime
import logging
import pickle

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env_power import PowerSystem
from gymnasium.wrappers import TimeLimit
from model_Hycon2 import get_cent_model, get_model_details
from mpcrl import Agent
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_power import plot_power_system_data
from scipy.linalg import block_diag

np.random.seed(1)

SCENARIO = True
PLOT = False
STORE_DATA = True

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
) = get_model_details()

num_scenarios = 100  # number of scenarios for scenario MPC


class ScenarioMpc(Mpc[cs.SX]):
    """A simple randomised scenario based MPC."""

    horizon = prediction_length

    Q_x_l = np.diag((500, 0.01, 0.01, 10))
    Q_x = block_diag(*([Q_x_l] * n))
    Q_u_l = 10
    Q_u = block_diag(*([Q_u_l] * n))

    def __init__(self) -> None:
        N = self.horizon
        gamma = discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        load = self.parameter("load", (n, 1))
        x_o = self.parameter("x_o", (n * nx_l, 1))
        u_o = self.parameter("u_o", (n * nu_l, 1))
        disturb = self.parameter("disturb", (n * num_scenarios, 1))

        self.fixed_parameter_dict = {
            "load": np.array([[0], [0], [0], [0]]),
            "x_o": np.zeros((n * nx_l, 1)),
            "u_o": np.zeros((n * nu_l, 1)),
            "disturb": np.random.uniform(
                -load_noise_bnd, load_noise_bnd, (n * num_scenarios, 1)
            ),
        }

        # action is normal for scenario
        u, _ = self.action("u", n * nu_l, lb=-u_lim, ub=u_lim)

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
            (n * num_scenarios, N),
            lb=0,
        )  # n in first dim as only cnstr on theta

        A, B, L = get_cent_model(discrete=True)

        # create dynamics matrices combining all scenarios
        A_full = block_diag(*[A] * num_scenarios)
        B_full = np.tile(B, (num_scenarios, 1))
        load_full = cs.repmat(load, num_scenarios, 1) + disturb
        L_full = block_diag(*[L] * num_scenarios)

        self.set_dynamics(
            lambda x, u: A_full @ x + B_full @ u + L_full @ load_full, n_in=2, n_out=1
        )

        # state constraints # TODO check the indexing here
        for i in range(num_scenarios):
            for j in range(n):  # only a constraint on theta
                for k in range(1, N):
                    self.constraint(
                        f"theta_lb_{i}_{j}_{k}",
                        -theta_lim - s[j + i * n, k],
                        "<=",
                        x[j * nx_l + i * n * nx_l, k],
                    )
                    self.constraint(
                        f"theta_ub_{i}_{j}_{k}",
                        x[j * nx_l + i * n * nx_l, k],
                        "<=",
                        theta_lim + s[j + i * n, k],
                    )

        # create cost matrices combining all scenarios
        Q_x_full = block_diag(*[self.Q_x] * num_scenarios)
        Q_u_full = num_scenarios * self.Q_u
        x_o_full = cs.repmat(x_o, num_scenarios, 1)
        w_full = np.tile(w, (num_scenarios, 1))
        self.minimize(
            +sum(
                +(gamma**k)
                * (
                    (x[:, k] - x_o_full).T @ Q_x_full @ (x[:, k] - x_o_full)
                    + (u[:, k] - u_o).T @ Q_u_full @ (u[:, k] - u_o)
                    + w_full.T @ s[:, [k]]
                )
                for k in range(N)
            )
            + (gamma**N) * (x[:, N] - x_o_full).T @ Q_x_full @ (x[:, N] - x_o_full)
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


class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC."""

    horizon = prediction_length

    Q_x_l = np.diag((500, 0.01, 0.01, 10))
    Q_x = block_diag(*([Q_x_l] * n))
    Q_u_l = 10
    Q_u = block_diag(*([Q_u_l] * n))

    def __init__(self) -> None:
        N = self.horizon
        gamma = discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        load = self.parameter("load", (n, 1))
        x_o = self.parameter("x_o", (n * nx_l, 1))
        u_o = self.parameter("u_o", (n * nu_l, 1))

        self.fixed_parameter_dict = {
            "load": np.array([[0], [0], [0], [0]]),
            "x_o": np.zeros((n * nx_l, 1)),
            "u_o": np.zeros((n * nu_l, 1)),
        }

        # action is normal for scenario
        u, _ = self.action("u", n * nu_l, lb=-u_lim, ub=u_lim)
        x, _ = self.state("x", n * nx_l)
        s, _, _ = self.variable(
            "s",
            (n, N),
            lb=0,
        )  # n in first dim as only cnstr on theta

        # state constraints

        for i in range(n):  # only a constraint on theta
            for k in range(1, N):
                self.constraint(
                    f"theta_lb_{i}_{k}",
                    -theta_lim - s[i, k],
                    "<=",
                    x[i * nx_l, k],
                )
                self.constraint(
                    f"theta_ub_{i}_{k}",
                    x[i * nx_l, k],
                    "<=",
                    theta_lim + s[i, k],
                )

        A, B, L = get_cent_model(discrete=True)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B @ u + L @ load, n_in=2, n_out=1)

        # objective
        self.minimize(
            +sum(
                +(gamma**k)
                * (
                    (x[:, k] - x_o).T @ self.Q_x @ (x[:, k] - x_o)
                    + (u[:, k] - u_o).T @ self.Q_u @ (u[:, k] - u_o)
                    + w.T @ s[:, [k]]
                )
                for k in range(N)
            )
            + (gamma**N) * (x[:, N] - x_o).T @ self.Q_x @ (x[:, N] - x_o)
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


class LoadedAgent(Agent):
    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o
        self.fixed_parameters["disturb"] = np.random.uniform(
            -load_noise_bnd, load_noise_bnd, (n * num_scenarios, 1)
        )
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env, episode: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o
        return super().on_episode_start(env, episode)


env = PowerSystem()
num_eps = 100
ep_len = 100
env = MonitorEpisodes(TimeLimit(PowerSystem(), max_episode_steps=int(ep_len)))

if SCENARIO:
    mpc = ScenarioMpc()
else:
    mpc = LinearMpc()
agent = Log(
    LoadedAgent(mpc, fixed_parameters=mpc.fixed_parameter_dict),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)
returns = agent.evaluate(env=env, episodes=num_eps, seed=1)

print(f"Total returns = {sum(returns)}")

# extract data
if len(env.observations) > 0:
    X = np.hstack([env.observations[i].squeeze().T for i in range(num_eps)]).T
    U = np.hstack([env.actions[i].squeeze().T for i in range(num_eps)]).T
    R = np.hstack([env.rewards[i].squeeze().T for i in range(num_eps)]).T
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

R_eps = [sum((R[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]
param_dict = {}
time = np.arange(R.size)
TD = []
TD_eps = []

if PLOT:
    plot_power_system_data(TD, R, TD_eps, R_eps, X, U)
if STORE_DATA:
    with open(
        "data/power_eval_S_"
        + str(SCENARIO)
        + datetime.datetime.now().strftime("%d%H%M%S%f")
        + str(".pkl"),
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)
