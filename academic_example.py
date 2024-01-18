import contextlib
import datetime
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt

# import networkx as netx
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm

CENTRALISED = False

Adj = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32
)  # adjacency matrix of coupling in network
G = g_map(Adj)  # mapping from global var to local var indexes for ADMM


def get_centralized_dynamics(
    n: int,
    nx_l: int,
    A_l: Union[cs.DM, cs.SX],
    B_l: Union[cs.DM, cs.SX],
    A_c: npt.NDArray[np.floating],
) -> tuple[Union[cs.DM, cs.SX], Union[cs.DM, cs.SX]]:
    """Creates the centralized representation of the dynamics from the real dynamics."""
    A = cs.SX.zeros(n * nx_l, n * nx_l)  # global state-space matrix A
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A[nx_l * i : nx_l * (i + 1), nx_l * i : nx_l * (i + 1)] = A_l
            elif Adj[i, j] == 1:
                A[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = A_c
                A[nx_l * j : nx_l * (j + 1), nx_l * i : nx_l * (i + 1)] = A_c
    with contextlib.suppress(RuntimeError):
        A = cs.evalf(A).full()
    B = cs.diagcat(*(B_l for _ in range(n)))  # global state-space matix B
    with contextlib.suppress(RuntimeError):
        B = cs.evalf(B).full()
    return A, B


def get_learnable_centralized_dynamics(
    n: int,
    nx_l: int,
    nu_l: int,
    A_list: List[Union[cs.DM, cs.SX]],
    B_list: List[Union[cs.DM, cs.SX]],
    A_c_list: List[List[npt.NDArray[np.floating]]],
    B_c_list: List[List[npt.NDArray[np.floating]]],
) -> tuple[Union[cs.DM, cs.SX], Union[cs.DM, cs.SX]]:
    """Creates the centralized representation of the dynamics from the learnable dynamics."""
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
    with contextlib.suppress(RuntimeError):
        A = cs.evalf(A).full()
    with contextlib.suppress(RuntimeError):
        B = cs.evalf(B).full()
    return A, B


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
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
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
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""
        action = action.full()
        x_new = self.A @ self.x + self.B @ action

        noise = self.np_random.uniform(*self.e_bnd).reshape(-1, 1)
        x_new[np.arange(0, self.nx, self.nx_l)] += noise

        r = self.get_stage_cost(self.x, action)
        r_dist = self.get_dist_stage_cost(self.x, action)
        self.x = x_new

        return x_new, r, False, False, {"r_dist": r_dist}


A_l_init = np.asarray([[1, 0.25], [0, 1]])
B_l_init = np.asarray([[0.0312], [0.25]])
A_c_l_init = np.array([[0, 0], [0, 0]])
B_c_l_init = np.array([[0], [0]])
learnable_pars_init_single = {
    "V0": np.zeros((1, 1)),
    "x_lb": np.reshape([0, 0], (-1, 1)),
    "x_ub": np.reshape([1, 0], (-1, 1)),
    "b": np.zeros(LtiSystem.nx_l),
    "f": np.zeros(LtiSystem.nx_l + LtiSystem.nu_l),
}


class LinearMpc(Mpc[cs.SX]):
    """The centralised MPC controller."""

    horizon = 10
    discount_factor = 0.9

    A_init, B_init = get_centralized_dynamics(
        LtiSystem.n, LtiSystem.nx_l, A_l_init, B_l_init, A_c_l_init
    )

    # add the initial guesses of the learable parameters
    learnable_pars_init = {}
    for i in range(LtiSystem.n):
        for name, val in learnable_pars_init_single.items():
            learnable_pars_init[f"{name}_{i}"] = val
        learnable_pars_init["A_" + str(i)] = A_l_init
        learnable_pars_init["B_" + str(i)] = B_l_init
        for j in range(LtiSystem.n):
            if i != j:
                if Adj[i][j] == 1:
                    learnable_pars_init["A_c_" + str(i) + "_" + str(j)] = A_c_l_init

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = LtiSystem.w
        nx, nu = LtiSystem.nx, LtiSystem.nu
        x_bnd, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # learn parameters with topology knowledge
        A_list = []
        B_list = []
        V0_list = []
        x_lb_list = []
        x_ub_list = []
        b_list = []
        f_list = []
        A_c_list: List[list] = []
        B_c_list: List[list] = []
        for i in range(LtiSystem.n):
            V0_list.append(self.parameter(f"V0_{i}", (1,)))
            x_lb_list.append(self.parameter(f"x_lb_{i}", (LtiSystem.nx_l,)))
            x_ub_list.append(self.parameter(f"x_ub_{i}", (LtiSystem.nx_l,)))
            b_list.append(self.parameter(f"b_{i}", (LtiSystem.nx_l, 1)))
            f_list.append(
                self.parameter(f"f_{i}", (LtiSystem.nx_l + LtiSystem.nu_l, 1))
            )

            A_list.append(
                self.parameter("A_" + str(i), (LtiSystem.nx_l, LtiSystem.nx_l))
            )
            B_list.append(
                self.parameter("B_" + str(i), (LtiSystem.nx_l, LtiSystem.nu_l))
            )

            A_c_list.append([])
            B_c_list.append([])
            for j in range(LtiSystem.n):
                A_c_list[i].append(None)
                B_c_list[i].append(None)
                if i != j:
                    if Adj[i][j] == 1:
                        A_c_list[i][j] = self.parameter(
                            "A_c_" + str(i) + "_" + str(j),
                            (LtiSystem.nx_l, LtiSystem.nx_l),
                        )
        # add listed params to one for intiialisation

        V0 = cs.vcat(V0_list)
        x_lb = cs.vcat(x_lb_list)
        x_ub = cs.vcat(x_ub_list)
        b = cs.vcat(b_list)
        f = cs.vcat(f_list)

        A, B = get_learnable_centralized_dynamics(
            LtiSystem.n,
            LtiSystem.nx_l,
            LtiSystem.nu_l,
            A_list,
            B_list,
            A_c_list,
            B_c_list,
        )

        # variables (state, action, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action(
            "u", nu, lb=a_bnd[0].reshape(-1, 1), ub=a_bnd[1].reshape(-1, 1)
        )
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B @ u + b, n_in=2, n_out=1)

        # other constraints
        self.constraint("x_lb", x_bnd[0].reshape(-1, 1) + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1].reshape(-1, 1) + x_ub + s)

        # objective
        S = cs.DM(
            dlqr(self.A_init, self.B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1]
        )
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            cs.sum1(V0)
            # + quad_form(S, x[:, -1])   # TODO I took this out for now cause we cant do it distributed
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
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
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class LocalMpc(MpcAdmm):
    """MPC for agent inner prob in ADMM."""

    rho = 0.5

    horizon = 10
    discount_factor = 0.9

    A_init = np.asarray([[1, 0.25], [0, 1]])
    B_init = np.asarray([[0.0312], [0.25]])
    A_c_l_init = np.array([[0, 0], [0, 0]])

    # learnable pars no related to coupling

    def __init__(self, num_neighbours, my_index) -> None:
        """Instantiate inner MPC for admm. My index is used to pick out own state from the grouped coupling states. It should be passed in via the mapping G (G[i].index(i))"""
        N = self.horizon
        gamma = self.discount_factor
        w_full = LtiSystem.w
        nx_l, nu_l = LtiSystem.nx_l, LtiSystem.nu_l
        w = w_full[:, 0:nx_l].reshape(-1, 1)
        x_bnd_full, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
        x_bnd = (
            x_bnd_full[0, 0:nx_l].reshape(-1, 1),
            x_bnd_full[1, 0:nx_l].reshape(-1, 1),
        )  # here assumed bounds are homogenous
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # learnable pars

        self.learnable_pars_init = {
            "V0": np.zeros((1, 1)),
            "x_lb": np.array([0, 0]).reshape(-1, 1),
            "x_ub": np.array([1, 0]).reshape(-1, 1),
            "b": np.zeros(LtiSystem.nx_l),
            "f": np.zeros(LtiSystem.nx_l + LtiSystem.nu_l),
            "A": self.A_init,
            "B": self.B_init,
        }

        self.num_neighbours = num_neighbours
        for i in range(
            num_neighbours
        ):  # add a learnable param init for each neighbours coupling matrix
            self.learnable_pars_init["A_c_" + str(i)] = self.A_c_l_init

        # fixed pars
        # parameters

        V0 = self.parameter("V0", (1,))
        x_lb = self.parameter("x_lb", (nx_l,))
        x_ub = self.parameter("x_ub", (nx_l,))
        b = self.parameter("b", (nx_l, 1))
        f = self.parameter("f", (nx_l + nu_l, 1))
        A = self.parameter("A", (nx_l, nx_l))
        B = self.parameter("B", (nx_l, nu_l))
        A_c_list: list[np.ndarray] = []  # list of coupling matrices
        for i in range(num_neighbours):
            A_c_list.append(self.parameter("A_c_" + str(i), (nx_l, nx_l)))

        # variables (state+coupling, action, slack)

        x, x_c = self.augmented_state(num_neighbours, my_index, nx_l)

        # TODO here assumed action bounds are homogenous
        u, _ = self.action(
            "u",
            nu_l,
            lb=a_bnd[0][0],
            ub=a_bnd[1][0],
        )
        s, _, _ = self.variable("s", (nx_l, N), lb=0)

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
                A @ x[:, [k]] + B @ u[:, [k]] + coup + b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints

        self.constraint(f"x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.set_local_cost(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
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


# now, let's create the instances of such classes and start the training
# centralised mpc and params
mpc = LinearMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)

# distributed mpc and params
mpc_dist_list: list[Mpc] = []
learnable_dist_parameters_list: list[LearnableParametersDict] = []
fixed_dist_parameters_list: list = []
for i in range(LtiSystem.n):
    mpc_dist_list.append(LocalMpc(num_neighbours=len(G[i]) - 1, my_index=G[i].index(i)))
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


env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=int(20e0)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgentCoordinator(
            rho=LocalMpc.rho,
            n=LtiSystem.n,
            G=G,
            Adj=Adj,
            centralised_flag=CENTRALISED,
            centralised_debug=True,
            mpc_cent=mpc,
            learnable_parameters=learnable_pars,
            mpc_dist_list=mpc_dist_list,
            learnable_dist_parameters_list=learnable_dist_parameters_list,
            fixed_dist_parameters_list=fixed_dist_parameters_list,
            discount_factor=mpc.discount_factor,
            update_strategy=2,
            learning_rate=ExponentialScheduler(6e-5, factor=0.9996),
            hessian_type="none",
            record_td_errors=True,
            exploration=EpsilonGreedyExploration(  # None,
                epsilon=ExponentialScheduler(0.7, factor=0.99),
                strength=0.5 * (LtiSystem.a_bnd[1, 0] - LtiSystem.a_bnd[0, 0]),
                seed=1,
            ),
            experience=ExperienceReplay(  # None,
                maxlen=100, sample_size=15, include_latest=10, seed=1
            ),  # None,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 200},
)

agent.train(env=env, episodes=1, seed=1)

STORE_DATA = False
PLOT = True

# extract data
if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)
TD = (
    agent.td_errors if CENTRALISED else agent.agents[0].td_errors
)  # all smaller agents have global TD error
time = np.arange(R.size)

# parameters
n = 3  # TODO remove hard coded n
updates = (
    np.arange(len(agent.updates_history["b_0"]))
    if CENTRALISED
    else np.arange(len(agent.agents[0].updates_history["b"]))
)
b = (
    [np.asarray(agent.updates_history[f"b_{i}"]) for i in range(LtiSystem.n)]
    if CENTRALISED
    else [
        np.asarray(agent.agents[i].updates_history["b"])
        for i in range(len(agent.agents))
    ]
)
f = (
    [np.asarray(agent.updates_history[f"f_{i}"]) for i in range(LtiSystem.n)]
    if CENTRALISED
    else [
        np.asarray(agent.agents[i].updates_history["f"])
        for i in range(len(agent.agents))
    ]
)
V0 = (
    [np.asarray(agent.updates_history[f"V0_{i}"]) for i in range(LtiSystem.n)]
    if CENTRALISED
    else [
        np.asarray(agent.agents[i].updates_history["V0"])
        for i in range(len(agent.agents))
    ]
)
bounds = (
    [
        np.concatenate(
            [np.squeeze(agent.updates_history[n]) for n in (f"x_lb_{i}", f"x_ub_{i}")],
            -1,
        )
        for i in range(LtiSystem.n)
    ]
    if CENTRALISED
    else [
        np.concatenate(
            [np.squeeze(agent.agents[i].updates_history[n]) for n in ("x_lb", "x_ub")],
            -1,
        )
        for i in range(len(agent.agents))
    ]
)
A = (
    [
        np.asarray(agent.updates_history[f"A_{i}"]).reshape(updates.size, -1)
        for i in range(n)
    ]  # TODO remove hard coded 3 agents
    if CENTRALISED
    else [
        np.asarray(agent.agents[i].updates_history["A"]).reshape(updates.size, -1)
        for i in range(len(agent.agents))
    ]
)
B = (
    [
        np.asarray(agent.updates_history[f"B_{i}"]).reshape(updates.size, -1)
        for i in range(n)
    ]  # TODO remove hard coded 3 agents
    if CENTRALISED
    else [
        np.asarray(agent.agents[i].updates_history["B"]).reshape(updates.size, -1)
        for i in range(len(agent.agents))
    ]
)
A_cs: list = []
for i in range(n):
    count = 0
    for j in range(n):
        if Adj[i, j] == 1:
            if CENTRALISED:
                A_cs.append(
                    np.asarray(agent.updates_history[f"A_c_{i}_{j}"]).reshape(
                        updates.size, -1
                    )
                )
            else:
                A_cs.append(
                    np.asarray(agent.agents[i].updates_history[f"A_c_{count}"]).reshape(
                        updates.size, -1
                    )
                )
                count += 1

# store data

if STORE_DATA:
    with open(
        "data/C_"
        + str(CENTRALISED)
        + datetime.datetime.now().strftime("%d%H%M%S%f")
        + str(".pkl"),
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(b, file)
        pickle.dump(f, file)
        pickle.dump(V0, file)
        pickle.dump(bounds, file)
        pickle.dump(A, file)
        pickle.dump(B, file)
        pickle.dump(A_cs, file)

if PLOT:
    # plot the results
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    axs[0].plot(time, X[:-1, np.arange(0, env.nx, env.nx_l)])
    axs[1].plot(time, X[:-1, np.arange(1, env.nx, env.nx_l)])
    axs[2].plot(time, U)
    for i in range(2):
        axs[0].axhline(env.x_bnd[i][0], color="r")
        axs[1].axhline(env.x_bnd[i][1], color="r")
        axs[2].axhline(env.a_bnd[i][0], color="r")
    axs[0].set_ylabel("$s_1$")
    axs[1].set_ylabel("$s_2$")
    axs[2].set_ylabel("$a$")

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(TD, "o", markersize=1)
    axs[1].semilogy(R, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    # Plot parameters
    _, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    for b_i in b:
        axs[0, 0].plot(updates, b_i)
    for bnd_i in bounds:
        axs[0, 1].plot(updates, bnd_i)
    for f_i in f:
        axs[1, 0].plot(updates, f_i)
    for V0_i in V0:
        axs[1, 1].plot(updates, V0_i.squeeze())
    for A_i in A:
        axs[2, 0].plot(updates, A_i)
    for B_i in B:
        axs[2, 1].plot(updates, B_i)

    axs[0, 0].set_ylabel("$b$")
    axs[0, 1].set_ylabel("$x_1$")
    axs[1, 0].set_ylabel("$f$")
    axs[1, 1].set_ylabel("$V_0$")
    axs[2, 0].set_ylabel("$A$")
    axs[2, 1].set_ylabel("$B$")
plt.show()
