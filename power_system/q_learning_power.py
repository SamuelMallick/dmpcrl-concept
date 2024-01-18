import datetime
import logging
import pickle

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from csnlp.wrappers.wrapper import Nlp
from env_power import PowerSystem
from gymnasium.wrappers import TimeLimit
from model_Hycon2 import (
    get_learnable_dynamics,
    get_learnable_dynamics_local,
    get_learned_P_tie_init,
    get_learned_pars_init_list,
    get_model_details,
    get_P_tie_init,
    get_pars_init_list,
)
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_power import plot_power_system_data

from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm

np.random.seed(1)

CENTRALISED = False
LEARN = True
USE_LEARNED_PARAMS = False

STORE_DATA = True
PLOT = False

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

w_l = w[0]  # agent penalty on state viols
b_scaling = 0.1  # scale the learnable model offset to prevent instability

# matrices for distributed conensus and ADMM
G = g_map(Adj)


class LocalMpc(MpcAdmm):
    """MPC for agent inner prob in ADMM."""

    rho = 50

    horizon = prediction_length

    # define learnable parameters

    to_learn = []
    # to_learn = to_learn + ["H"]
    # to_learn = to_learn + ["D"]
    # to_learn = to_learn + ["T_t"]
    # to_learn = to_learn + ["T_g"]
    to_learn = to_learn + ["theta_lb"]
    to_learn = to_learn + ["theta_ub"]
    to_learn = to_learn + ["V0"]
    to_learn = to_learn + ["b"]
    to_learn = to_learn + ["f_x"]
    to_learn = to_learn + ["f_u"]
    to_learn = to_learn + ["Q_x"]
    to_learn = to_learn + ["Q_u"]

    def __init__(self, num_neighbours, my_index, pars_init, P_tie_init, u_lim) -> None:
        """Instantiate inner MPC for admm. My index is used to pick out own state from the grouped coupling states. It should be passed in via the mapping G (G[i].index(i))"""

        # add coupling to learn
        for i in range(num_neighbours):
            self.to_learn = self.to_learn + [f"P_tie_{i}"]

        N = self.horizon
        gamma = discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # init param vals

        self.learnable_pars_init = {}
        self.fixed_pars_init["load"] = np.zeros((n, 1))
        self.fixed_pars_init["x_o"] = np.zeros((n * nx_l, 1))
        self.fixed_pars_init["u_o"] = np.zeros((n * nu_l, 1))

        # model params
        for name, val in pars_init.items():
            if name in self.to_learn:
                self.learnable_pars_init[name] = val
            else:
                self.fixed_pars_init[name] = val
        for i in range(num_neighbours):
            if f"P_tie_{i}" in self.to_learn:
                self.learnable_pars_init[f"P_tie_{i}"] = P_tie_init[i]
            else:
                self.fixed_pars_init[f"P_tie_{i}"] = P_tie_init[i]

        # create the params

        H = self.parameter("H", (1,))
        R = self.parameter("R", (1,))
        D = self.parameter("D", (1,))
        T_t = self.parameter("T_t", (1,))
        T_g = self.parameter("T_g", (1,))
        P_tie_list = []
        for i in range(num_neighbours):
            P_tie_list.append(self.parameter(f"P_tie_{i}", (1,)))

        theta_lb = self.parameter(f"theta_lb", (1,))
        theta_ub = self.parameter(f"theta_ub", (1,))

        V0 = self.parameter(f"V0", (1,))
        b = self.parameter(f"b", (nx_l,))
        f_x = self.parameter(f"f_x", (nx_l,))
        f_u = self.parameter(f"f_u", (nu_l,))
        Q_x = self.parameter(f"Q_x", (nx_l, nx_l))
        Q_u = self.parameter(f"Q_u", (nu_l, nu_l))

        load = self.parameter("load", (1, 1))
        x_o = self.parameter("x_o", (nx_l, 1))
        u_o = self.parameter("u_o", (nu_l, 1))

        A, B, L, A_c_list = get_learnable_dynamics_local(H, R, D, T_t, T_g, P_tie_list)

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)

        u, _ = self.action(
            "u",
            nu_l,
            lb=-u_lim,
            ub=u_lim,
        )
        s, _, _ = self.variable("s", (1, N), lb=0)  # dim 1 as only theta has bound

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        # dynamics - added manually due to coupling
        for k in range(N):
            coup = cs.SX.zeros(nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                "dynam_" + str(k),
                A @ x[:, [k]] + B @ u[:, [k]] + L @ load + coup + b_scaling * b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints

        self.constraint(f"theta_lb", -theta_lim + theta_lb - s, "<=", x[0, 1:])
        self.constraint(f"theta_ub", x[0, 1:], "<=", theta_lim + theta_ub + s)

        # objective
        self.set_local_cost(
            V0
            + sum(
                f_x.T @ x[:, k]
                + f_u.T @ u[:, k]
                + (gamma**k)
                * (
                    (x[:, k] - x_o).T @ Q_x @ (x[:, k] - x_o)
                    + (u[:, k] - u_o).T @ Q_u @ (u[:, k] - u_o)
                    + w_l * s[:, [k]]
                )
                for k in range(N)
            )
        )

        self.nx_l = nx_l
        self.nu_l = nu_l

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
                "max_iter": 2000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class CentralisedMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = prediction_length

    # define which params are learnable
    to_learn = []
    # to_learn = [f"H_{i}" for i in range(n)]
    # to_learn = to_learn + [f"R_{i}" for i in range(n)]
    # to_learn = to_learn + [f"D_{i}" for i in range(n)]
    # to_learn = to_learn + [f"T_t_{i}" for i in range(n)]
    # to_learn = to_learn + [f"T_g_{i}" for i in range(n)]
    to_learn = to_learn + [f"theta_lb_{i}" for i in range(n)]
    to_learn = to_learn + [f"theta_ub_{i}" for i in range(n)]
    to_learn = to_learn + [f"V0_{i}" for i in range(n)]
    to_learn = to_learn + [f"b_{i}" for i in range(n)]
    to_learn = to_learn + [f"f_x_{i}" for i in range(n)]
    to_learn = to_learn + [f"f_u_{i}" for i in range(n)]
    to_learn = to_learn + [f"Q_x_{i}" for i in range(n)]
    to_learn = to_learn + [f"Q_u_{i}" for i in range(n)]
    to_learn = to_learn + [
        f"P_tie_{i}_{j}" for j in range(n) for i in range(n) if Adj[i, j] == 1
    ]

    # initialise parameters vals

    learnable_pars_init = {}
    fixed_pars_init = {
        "load": np.zeros((n, 1)),
        "x_o": np.zeros((n * nx_l, 1)),
        "u_o": np.zeros((n * nu_l, 1)),
    }

    # model params
    if USE_LEARNED_PARAMS:
        P_tie_init = get_learned_P_tie_init()
        pars_init_list = get_learned_pars_init_list()
    else:
        P_tie_init = get_P_tie_init()
        pars_init_list = get_pars_init_list()
    for i in range(n):
        for name, val in pars_init_list[i].items():
            if f"{name}_{i}" in to_learn:
                learnable_pars_init[f"{name}_{i}"] = val
            else:
                fixed_pars_init[f"{name}_{i}"] = val

    # coupling params
    for i in range(n):
        for j in range(n):
            if Adj[i, j] == 1:
                if f"P_tie_{i}_{j}" in to_learn:
                    learnable_pars_init[f"P_tie_{i}_{j}"] = P_tie_init[i, j]
                else:
                    fixed_pars_init[f"P_tie_{i}_{j}"] = P_tie_init[i, j]

    def __init__(self) -> None:
        N = self.horizon
        gamma = discount_factor
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # init params

        H_list = [self.parameter(f"H_{i}", (1,)) for i in range(n)]
        R__list = [self.parameter(f"R_{i}", (1,)) for i in range(n)]
        D_list = [self.parameter(f"D_{i}", (1,)) for i in range(n)]
        T_t_list = [self.parameter(f"T_t_{i}", (1,)) for i in range(n)]
        T_g_list = [self.parameter(f"T_g_{i}", (1,)) for i in range(n)]
        P_tie_list_list = []
        for i in range(n):
            P_tie_list_list.append([])
            for j in range(n):
                if Adj[i, j] == 1:
                    P_tie_list_list[i].append(self.parameter(f"P_tie_{i}_{j}", (1,)))
                else:
                    P_tie_list_list[i].append(0)

        A, B, L = get_learnable_dynamics(
            H_list, R__list, D_list, T_t_list, T_g_list, P_tie_list_list
        )

        load = self.parameter("load", (n, 1))
        x_o = self.parameter("x_o", (n * nx_l, 1))
        u_o = self.parameter("u_o", (n * nu_l, 1))

        theta_lb = [self.parameter(f"theta_lb_{i}", (1,)) for i in range(n)]
        theta_ub = [self.parameter(f"theta_ub_{i}", (1,)) for i in range(n)]

        V0 = [self.parameter(f"V0_{i}", (1,)) for i in range(n)]
        b = [self.parameter(f"b_{i}", (nx_l,)) for i in range(n)]
        f_x = [self.parameter(f"f_x_{i}", (nx_l,)) for i in range(n)]
        f_u = [self.parameter(f"f_u_{i}", (nu_l,)) for i in range(n)]
        Q_x = [self.parameter(f"Q_x_{i}", (nx_l, nx_l)) for i in range(n)]
        Q_u = [self.parameter(f"Q_u_{i}", (nu_l, nu_l)) for i in range(n)]

        # mpc vars

        x, _ = self.state("x", n * nx_l)
        u, _ = self.action("u", n * nu_l, lb=-u_lim, ub=u_lim)
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
                    -theta_lim + theta_lb[i] - s[i, k],
                    "<=",
                    x[i * nx_l, k],
                )
                self.constraint(
                    f"theta_ub_{i}_{k}",
                    x[i * nx_l, k],
                    "<=",
                    theta_lim + theta_ub[i] + s[i, k],
                )

        # dynamics

        b_full = cs.SX()
        for i in range(n):
            b_full = cs.vertcat(b_full, b[i])
        self.set_dynamics(
            lambda x, u: A @ x + B @ u + L @ load + b_scaling * b_full, n_in=2, n_out=1
        )

        # objective

        Q_x_full = cs.diagcat(*Q_x)
        Q_u_full = cs.diagcat(*Q_u)

        f_x_full = cs.SX()
        f_u_full = cs.SX()
        for i in range(n):
            f_x_full = cs.vertcat(f_x_full, f_x[i])
            f_u_full = cs.vertcat(f_u_full, f_u[i])

        self.minimize(
            sum(V0)
            + sum(
                f_x_full.T @ x[:, k]
                + f_u_full.T @ u[:, k]
                + (gamma**k)
                * (
                    (x[:, k] - x_o).T @ Q_x_full @ (x[:, k] - x_o)
                    + (u[:, k] - u_o).T @ Q_u_full @ (u[:, k] - u_o)
                    + w.T @ s[:, [k]]
                )
                for k in range(N)
            )
            + (gamma**N) * (x[:, N] - x_o).T @ Q_x_full @ (x[:, N] - x_o)
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


# override the learning agent to check for new load values each iter
class LoadedLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o

        if not self.centralised_flag:
            for i in range(n):
                self.agents[i].fixed_parameters["load"] = env.load[i]
                self.agents[i].fixed_parameters["x_o"] = env.x_o[
                    nx_l * i : nx_l * (i + 1)
                ]
                self.agents[i].fixed_parameters["u_o"] = env.u_o[i]

        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env, episode: int) -> None:
        self.fixed_parameters["load"] = env.load
        self.fixed_parameters["x_o"] = env.x_o
        self.fixed_parameters["u_o"] = env.u_o

        if not self.centralised_flag:
            for i in range(n):
                self.agents[i].fixed_parameters["load"] = env.load[i]
                self.agents[i].fixed_parameters["x_o"] = env.x_o[
                    nx_l * i : nx_l * (i + 1)
                ]
                self.agents[i].fixed_parameters["u_o"] = env.u_o[i]

        return super().on_episode_start(env, episode)


# create distributed mpc's and parameters
if USE_LEARNED_PARAMS:
    P_tie_init = get_learned_P_tie_init()
    pars_init_list = get_learned_pars_init_list()
else:
    P_tie_init = get_P_tie_init()
    pars_init_list = get_pars_init_list()
# distributed mpc and params
mpc_dist_list: list[Mpc] = []
learnable_dist_parameters_list: list[LearnableParametersDict] = []
fixed_dist_parameters_list: list = []

for i in range(n):
    mpc_dist_list.append(
        LocalMpc(
            num_neighbours=len(G[i]) - 1,
            my_index=G[i].index(i),
            pars_init=pars_init_list[i],
            P_tie_init=[P_tie_init[i, j] for j in range(n) if Adj[i, j] != 0],
            u_lim=u_lim[i],
        )
    )
    learnable_dist_parameters_list.append(
        LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name, val.shape, val, sym=mpc_dist_list[i].parameters[name]
                )
                for name, val in mpc_dist_list[i].learnable_pars_init.items()
            )
        )
    )
    fixed_dist_parameters_list.append(mpc_dist_list[i].fixed_pars_init)

# create distributed mpc's and parameters
mpc = CentralisedMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)
ep_len = int(100)
env = MonitorEpisodes(TimeLimit(PowerSystem(), max_episode_steps=int(ep_len)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LoadedLstdQLearningAgentCoordinator(
            rho=LocalMpc.rho,
            n=n,
            G=G,
            Adj=Adj,
            centralised_flag=CENTRALISED,
            centralised_debug=True,
            mpc_cent=mpc,
            learnable_parameters=learnable_pars,
            fixed_parameters=mpc.fixed_pars_init,
            mpc_dist_list=mpc_dist_list,
            learnable_dist_parameters_list=learnable_dist_parameters_list,
            fixed_dist_parameters_list=fixed_dist_parameters_list,
            discount_factor=discount_factor,
            update_strategy=ep_len,
            learning_rate=ExponentialScheduler(7e-7, factor=0.98),  # 5e-6
            hessian_type="none",
            record_td_errors=True,
            exploration=EpsilonGreedyExploration(  # None,
                epsilon=ExponentialScheduler(0.5, factor=0.8),
                strength=0.1 * (2 * 0.2),
                seed=1,
            ),
            experience=ExperienceReplay(  # None,
                maxlen=3 * ep_len,
                sample_size=int(1.5 * ep_len),
                include_latest=ep_len,
                seed=1,
            ),  # None,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

identifier = "line_40_with_con_eval"
num_eps = 100
if LEARN:
    agent.train(env=env, episodes=num_eps, seed=1)
else:
    agent.evaluate(env=env, episodes=num_eps, seed=1)

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
if LEARN:
    TD = np.squeeze(agent.td_errors) if CENTRALISED else agent.agents[0].td_errors
    TD_eps = [sum((TD[ep_len * i : ep_len * (i + 1)])) / ep_len for i in range(num_eps)]
    if CENTRALISED:
        for name in mpc.to_learn:
            param_dict[name] = np.asarray(agent.updates_history[name])
    else:
        for i in range(n):
            for name in mpc_dist_list[i].to_learn:
                param_dict[name + "_" + str(i)] = np.asarray(
                    agent.agents[i].updates_history[name]
                )

if STORE_DATA:
    with open(
        "data/power_C_"
        + str(CENTRALISED)
        + identifier
        + datetime.datetime.now().strftime("%d%H%M%S%f")
        + str(".pkl"),
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)

if PLOT:
    plot_power_system_data(
        TD, R, TD_eps, R_eps, X, U, param_dict=param_dict if LEARN else None
    )
