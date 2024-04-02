import pickle

import matplotlib.pyplot as plt
from dmpcrl.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")
num_eps = 100
ep_len = 100
nx_l = 4
n = 4
theta_lim = 0.1

with open(
    "data/power_data/line_40/distributed_con_eval.pkl",
    "rb",
) as file:
    X_l = pickle.load(file)
    U_l = pickle.load(file)
    R_l = pickle.load(file)
    TD_l = pickle.load(file)
    param_list_l = pickle.load(file)

with open(
    "data/power_data/scenario/power_scenario_79.pkl",
    "rb",
) as file:
    X_s = pickle.load(file)
    U_s = pickle.load(file)
    R_s = pickle.load(file)
    TD_s = pickle.load(file)
    param_list_s = pickle.load(file)

with open(
    "data/power_data/nominal/centralised.pkl",
    "rb",
) as file:
    X_n = pickle.load(file)
    U_n = pickle.load(file)
    R_n = pickle.load(file)
    TD_n = pickle.load(file)
    param_list_n = pickle.load(file)

R_l_eps = [sum(R_l[ep_len * i : ep_len * (i + 1)]) for i in range(num_eps)]
R_s_eps = [sum(R_s[ep_len * i : ep_len * (i + 1)]) for i in range(num_eps)]
R_n_eps = [sum(R_n[ep_len * i : ep_len * (i + 1)]) for i in range(num_eps)]

# count violations per ep
V_l_eps = []
V_s_eps = []
V_n_eps = []
for i in range(num_eps):
    V_l_eps.append(0)
    V_s_eps.append(0)
    V_n_eps.append(0)
    for k in range(ep_len):
        for j in range(n):
            if (
                X_l[k + i * ep_len, j * nx_l] > theta_lim
                or X_l[k + i * ep_len, j * nx_l] < -theta_lim
            ):
                V_l_eps[i] += 1
                break

        for j in range(n):
            if (
                X_s[k + i * ep_len, j * nx_l] > theta_lim
                or X_s[k + i * ep_len, j * nx_l] < -theta_lim
            ):
                V_s_eps[i] += 1
                break

        for j in range(n):
            if (
                X_n[k + i * ep_len, j * nx_l] > theta_lim
                or X_n[k + i * ep_len, j * nx_l] < -theta_lim
            ):
                V_n_eps[i] += 1
                break
fig, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.boxplot(
    [V_l_eps, V_s_eps, V_n_eps], notch=False, labels=["policy", "stochastic", "nominal"]
)  # , positions=[0.3, 1.3, 2.3], widths=0.3)
axs.set_ylabel(r"\# cnstr violations")
fig.set_size_inches(4, 2)
plt.savefig("data/box2.svg", format="svg", dpi=300)

fig, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.boxplot(
    [R_l_eps, R_s_eps, R_n_eps],
    notch=False,
    labels=["policy", "stochastic", "nominal"],
    positions=[0, 1, 2],
    widths=0.3,
)
axs.set_ylabel(r"$\sum L$")
# axs = axs.twinx()
# axs.boxplot([V_l_eps, V_s_eps, V_n_eps], notch=False, labels=["distributed policy", "stochastic MPC", "nominal MPC"], positions=[0.3, 1.3, 2.3], widths=0.3)
fig.set_size_inches(4, 2)
plt.savefig("data/box.svg", format="svg", dpi=300)
save2tikz(plt.gcf())

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R_l_eps, "o", color="C0", markersize=1.5)
axs.plot(R_s_eps, "o", color="C1", markersize=1.5)
axs.plot(R_n_eps, "o", color="C2", markersize=1.5)
axs.axhline(sum(R_l_eps) / len(R_l_eps), linestyle="--", color="C0", linewidth=1)
axs.axhline(sum(R_s_eps) / len(R_s_eps), linestyle="--", color="C1", linewidth=1)
axs.axhline(sum(R_n_eps) / len(R_n_eps), linestyle="--", color="C2", linewidth=1)
axs.set_xlabel("episode")
axs.set_ylabel(r"$\sum L$")
axs.legend(["learned", "scenario", "nominal"])
plt.savefig("data/eval.svg", format="svg", dpi=300)
plt.show()
