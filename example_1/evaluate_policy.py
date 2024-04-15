import pickle

import casadi as cs

# import networkx as netx
from csnlp.wrappers import Mpc
from env import LtiSystem
from learnable_mpc import LocalMpc
from mpcrl import LearnableParameter, LearnableParametersDict
from model import get_adj, get_bounds, get_model_details
from dmpcrl.core.admm import g_map
import datetime
import logging
import pickle

import casadi as cs
import matplotlib.pyplot as plt

# import networkx as netx
import numpy as np
from csnlp.wrappers import Mpc
from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import g_map
from env import LtiSystem
from gymnasium.wrappers import TimeLimit
from learnable_mpc import CentralizedMpc, LocalMpc
from model import get_adj, get_bounds, get_model_details
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

Adj = get_adj()
G = g_map(Adj)  # mapping from global var to local var indexes for ADMM
rho = 0.5
n, nx_l, nu_l = get_model_details()
_, u_bnd, _ = get_bounds()

with open(
    "data/example_1/line_416/distributed.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    TD = pickle.load(file)
    b = pickle.load(file)
    f = pickle.load(file)
    V0 = pickle.load(file)
    bounds = pickle.load(file)
    A = pickle.load(file)
    B = pickle.load(file)
    A_cs = pickle.load(file)

policy_time_step = 7500
learned_pars: list[dict] = [{}, {}, {}]
couple_param_index = 0
for i in range(n):
    learned_pars[i]["b"] = b[i][policy_time_step].reshape(nx_l, 1)
    learned_pars[i]["f"] = f[i][policy_time_step].reshape(nx_l + nu_l, 1)
    learned_pars[i]["V0"] = V0[i][policy_time_step].reshape(
        1,
    )
    learned_pars[i]["x_lb"] = bounds[i][policy_time_step, :2].reshape(
        nx_l,
    )
    learned_pars[i]["x_ub"] = bounds[i][policy_time_step, 2:].reshape(
        nx_l,
    )
    learned_pars[i]["A"] = A[i][policy_time_step].reshape(nx_l, nx_l)
    learned_pars[i]["B"] = B[i][policy_time_step].reshape(nx_l, nu_l)
    for j in range(len(G[i]) - 1):  # number of neighbors
        learned_pars[i][f"A_c_{j}"] = A_cs[couple_param_index][policy_time_step].reshape(nx_l, nx_l)
        couple_param_index += 1

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
    mpc_dist_list.append(
        LocalMpc(num_neighbours=len(G[i]) - 1, my_index=G[i].index(i), rho=rho)
    )
    learnable_dist_parameters_list.append(
        LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name, val.shape, val, sym=mpc_dist_list[i].parameters[name]
                )
                for name, val in learned_pars[i].items()
            )
        )
    )
    fixed_dist_parameters_list.append(mpc_dist_list[i].fixed_pars_init)

env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=int(100)))
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgentCoordinator(
            rho=rho,
            n=n,
            G=G,
            Adj=Adj,
            centralised_flag=False,
            centralised_debug=False,
            mpc_cent=mpc,
            learnable_parameters=learnable_pars,
            mpc_dist_list=mpc_dist_list,
            learnable_dist_parameters_list=learnable_dist_parameters_list,
            fixed_dist_parameters_list=fixed_dist_parameters_list,
            discount_factor=mpc_dist_list[0].discount_factor,
            update_strategy=2,
            learning_rate=ExponentialScheduler(6e-5, factor=0.9996),
            hessian_type="none",
            record_td_errors=True,
            exploration=EpsilonGreedyExploration(
                epsilon=ExponentialScheduler(0.7, factor=0.99),
                strength=0.5 * (u_bnd[1, 0] - u_bnd[0, 0]),
                seed=1,
            ),
            experience=ExperienceReplay(
                maxlen=100, sample_size=15, include_latest=10, seed=1
            ),
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

agent.evaluate(env=env, episodes=1, seed=1)

# extract data
if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

with open(
    f"eval_pol_step_{policy_time_step}" + ".pkl",
    "wb",
) as file:
    pickle.dump(X, file)
    pickle.dump(U, file)
    pickle.dump(R, file)
