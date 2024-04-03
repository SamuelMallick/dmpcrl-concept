import datetime
import logging
import pickle

import casadi as cs
import matplotlib.pyplot as plt

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm
from env import LtiSystem
from gymnasium.wrappers import TimeLimit
from model import get_adj, get_centralized_dynamics, get_learnable_centralized_dynamics, get_model_details, get_bounds
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from learnable_mpc import CentralizedMpc, LocalMpc

CENTRALISED = False

Adj = get_adj()
G = g_map(Adj)  # mapping from global var to local var indexes for ADMM
rho = 0.5
n, nx_l, nu_l = get_model_details()
_, u_bnd, _ = get_bounds()

# now, let's create the instances of such classes and start the training
# centralised mpc and params
mpc = CentralizedMpc()
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
    mpc_dist_list.append(LocalMpc(num_neighbours=len(G[i]) - 1, my_index=G[i].index(i), rho=rho))
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
            rho=rho,
            n=n,
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
                strength=0.5 * (u_bnd[1, 0] - u_bnd[0, 0]),
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
        + ".pkl",
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
