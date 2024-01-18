import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from dmpcrl.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")
CENTRALISED = False
LEARNED = True
num_eps = 300
ep_len = 100
nx_l = 4
n = 4
theta_lim = 0.1
u_lim = np.array([[0.2], [0.1], [0.3], [0.1]])
P_tie = np.array(
    [
        [0, 4, 0, 0],
        [4, 0, 2, 0],
        [0, 2, 0, 2],
        [0, 0, 2, 0],
    ]
)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

with open(
    "data/power_data/line_40/distributed_con.pkl",
    # "data/power_data/scenario/power_scenario_79.pkl",
    # "data/power_data/nominal/centralised.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    TD = pickle.load(file)
    param_list = pickle.load(file)

# plot the results
TD_eps = [sum((TD[ep_len * i : ep_len * (i + 1)])) / ep_len for i in range(num_eps)]
R_eps = [sum((R[ep_len * i : ep_len * (i + 1)])) for i in range(num_eps)]

print(f"Average cost = {sum(R_eps)/len(R_eps)}")
# manually enter average cost of scenario and nominal MPC
av_cost_scen = 317.9
av_cost_nom = 349

fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD_eps, "o", color="C0", markersize=0.8)
axs[1].plot(R_eps, "o", color="C0", markersize=0.8)
# axs[1].axhline(av_cost_scen, color = 'blue', linewidth=1)
# axs[1].axhline(av_cost_nom, color = 'green', linewidth=1)
axs[0].set_ylabel(r"$\overline{\delta}$")
axs[1].set_ylabel(r"$\sum_{t = 0}^{100} L_t$")
axs[1].set_xlabel("episode")
fig.set_size_inches(8, 3.5)
plt.savefig("data/TD.svg", format="svg", dpi=300)

# calculate percentage of constraint violation
viol_count = 0
for i in range(X.shape[0]):
    for j in range(n):
        if X[i, j * nx_l] > theta_lim or X[i, j * nx_l] < -theta_lim:
            viol_count += 1
            break
viol_percent = (viol_count / X.shape[0]) * 100
print(f"Violated {viol_percent}% of time-steps")

# first episode
_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i in range(n):
    for j in range(nx_l):
        axs[j].plot(X[: ep_len + 1, i * nx_l + j], color=colors[i])
    axs[4].plot(U[:ep_len, i], color=colors[i])
    axs[4].axhline(u_lim[i], color=colors[i], linewidth=1, linestyle="--")
    axs[4].axhline(-u_lim[i], color=colors[i], linewidth=1, linestyle="--")

axs[0].axhline(theta_lim, color="r", linewidth=1)
axs[0].axhline(-theta_lim, color="r", linewidth=1)


# last episode
_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i in range(n):
    for j in range(nx_l):
        axs[j].plot(X[-ep_len - 1 :, i * nx_l + j], color=colors[i])
    axs[4].plot(U[-ep_len:, i], color=colors[i])
    axs[4].axhline(u_lim[i], color=colors[i], linewidth=1, linestyle="--")
    axs[4].axhline(-u_lim[i], color=colors[i], linewidth=1, linestyle="--")

axs[0].axhline(theta_lim, color="r", linewidth=1)
axs[0].axhline(-theta_lim, color="r", linewidth=1)

# all episodes
_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i in range(n):
    for j in range(nx_l):
        axs[j].plot(X[:, i * nx_l + j], color=colors[i])
    axs[4].plot(U[:, i], color=colors[i])
    axs[4].axhline(u_lim[i], color=colors[i], linewidth=1, linestyle="--")
    axs[4].axhline(-u_lim[i], color=colors[i], linewidth=1, linestyle="--")

axs[0].axhline(theta_lim, color="r", linewidth=1)
axs[0].axhline(-theta_lim, color="r", linewidth=1)

# tie line power flows
legend_entries = []
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
if True:
    for i in range(n):
        for j in range(n):
            if P_tie[i, j] != 0:
                # first ep
                axs[0].plot(
                    P_tie[i, j]
                    * (X[: ep_len + 1, i * nx_l] - X[: ep_len + 1, j * nx_l])
                )
                # last ep
                axs[1].plot(
                    P_tie[i, j]
                    * (X[-ep_len - 1 :, i * nx_l] - X[-ep_len - 1 :, j * nx_l])
                )
                legend_entries.append(f"{i+1}{j+1}")
    axs[0].legend(legend_entries)

# state and tie line plot for paper
fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[: ep_len + 1, 12], color="C0", linestyle="--", linewidth=0.8)
axs[0].plot(X[-ep_len - 1 :, 12], color="C0", linestyle="-", linewidth=0.8)
axs[0].axhline(theta_lim, color="r", linewidth=1)
axs[0].axhline(-theta_lim, color="r", linewidth=1)
axs[1].plot(
    P_tie[2, 3] * (X[: ep_len + 1, 8] - X[: ep_len + 1, 12]),
    color="C0",
    linestyle="--",
    linewidth=0.8,
)
axs[1].plot(
    P_tie[2, 3] * (X[-ep_len - 1 :, 8] - X[-ep_len - 1 :, 12]),
    color="C0",
    linestyle="-",
    linewidth=0.8,
)
axs[1].set_xlabel("$k$")
axs[0].set_ylabel(r"$\Delta \phi_4$")
axs[1].set_ylabel(r"$\Delta P_{tie,3,4}$")
fig.set_size_inches(8, 3.5)
plt.savefig("data/states.svg", format="svg", dpi=300)

# parameters
if LEARNED:
    _, axs = plt.subplots(4, 2, constrained_layout=True, sharex=True)
    idx = 2
    axs[0, 0].plot(param_list[f"theta_lb_{idx}"], color="C0", linewidth=0.6)
    axs[0, 0].set_ylabel(r"$\underline{\Delta \phi}_{3}, \overline{\Delta \phi}_{3}$")
    axs[0, 0].plot(param_list[f"theta_ub_{idx}"], color="C0", linewidth=0.6)
    axs[1, 0].plot(param_list[f"V0_{idx}"], color="C0", linewidth=0.6)
    axs[1, 0].set_ylabel("$V_{3,0}$")
    axs[2, 0].plot(param_list[f"b_{idx}"], color="C0", linewidth=0.6)
    axs[2, 0].set_ylabel("$b_3$")
    axs[3, 0].plot(
        param_list[f"f_x_{idx}"].reshape((num_eps + 1, 4)), color="C0", linewidth=0.6
    )
    axs[3, 0].set_ylabel("$f_3$")
    axs[3, 0].plot(
        param_list[f"f_u_{idx}"].reshape((num_eps + 1, 1)), color="C0", linewidth=0.6
    )
    axs[0, 1].plot(
        param_list[f"Q_x_{idx}"].reshape((num_eps + 1, 16)),
        color="C0",
        linewidth=0.6,
    )
    axs[0, 1].set_ylabel("$Q_{x_3}$")
    axs[1, 1].plot(
        param_list[f"Q_u_{idx}"].reshape((num_eps + 1, 1)), color="C0", linewidth=0.6
    )
    axs[1, 1].set_ylabel("$Q_{u_3}$")
    if CENTRALISED:
        axs[2, 1].plot(param_list[f"P_tie_{idx}_{idx-1}"], color="C0", linewidth=0.6)
        axs[2, 1].set_ylabel("$P_{3,2}$")
        axs[3, 1].plot(param_list[f"P_tie_{idx}_{idx+1}"], color="C0", linewidth=0.6)
        axs[3, 1].set_ylabel("$P_{3,4}$")
    else:
        axs[2, 1].plot(param_list[f"P_tie_0_{idx}"], color="C0", linewidth=0.6)
        axs[2, 1].set_ylabel("$P_{3,2}$")
        axs[3, 1].plot(param_list[f"P_tie_1_{idx}"], color="C0", linewidth=0.6)
        axs[3, 1].set_ylabel("$P_{3,4}$")
    plt.savefig("data/params.svg", format="svg", dpi=300)
plt.show()
