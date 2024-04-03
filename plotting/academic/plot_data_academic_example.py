import pickle

import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
from matplotlib.ticker import FormatStrFormatter

# plt.rcParams.update({
#   "text.usetex": True,
# })

plt.rc("text", usetex=True)
plt.rc("font", size=19)
plt.style.use("bmh")

nx = 6
nx_l = 2
x_bnd = np.array([[0, -1], [1, 1]])
a_bnd = np.array([[-1], [1]])
update_rate = 2

limit = 20000

with open(
    "data/academic_ex_data/line_416/distributed.pkl",
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

# reshape data to plot only averages
window_size = 10
TD_array = np.array(TD)
TD_reshaped = TD_array.reshape(-1, window_size)
# TD = np.mean(TD_reshaped, axis=1)  #Plot average of window size
# TD = pd.Series(TD).rolling(window=window_size).mean()  # Plot moving average
# x_pnts = np.arange(limit)
# x_pnts = x_pnts[::window_size]
# TD = TD[::window_size]  # plot every window_size'th point

# plot the results
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[:limit, np.arange(0, nx, nx_l)], linewidth=0.25, rasterized=True)
# axs[0].plot(X[:limit, [2, 4, 0]])
axs[1].plot(X[:limit, np.arange(1, nx, nx_l)], linewidth=0.25, rasterized=True)
# axs[1].plot(X[:limit, [3, 5, 1]])
# axs[2].plot(U[:limit, [1, 2, 0]])
axs[2].plot(U[:limit], linewidth=0.25, rasterized=True)
for i in range(2):
    axs[0].axhline(x_bnd[i][0], color="r", linewidth=2, linestyle="--")
    axs[1].axhline(x_bnd[i][1], color="r", linewidth=2, linestyle="--")
    axs[2].axhline(a_bnd[i][0], color="r", linewidth=2, linestyle="--")
# axs[0].set_ylabel("$s_1$")
# axs[1].set_ylabel("$s_2$")
# axs[2].set_ylabel("$a$")
# axs[2].set_xlabel(r"$t$")
# axs[0].set_yticklabels([])
# axs[1].set_yticklabels([])
# axs[2].set_yticklabels([])
# axs[2].set_xlabel(r"$t$")
plt.savefig("data/states.svg", format="svg", dpi=300)

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD[:limit], "o", color="C0", markersize=0.8, rasterized=True)
axs[1].plot(R[:limit], "o", color="C0", markersize=0.8, rasterized=True)
axs[0].set_ylabel(r"$\delta$")
axs[1].set_ylabel(r"$L$")
axs[1].set_xlabel(r"$t$")
axs[0].set_ylim(-5, 15)
axs[1].set_ylim(0, 8)
# axs[1].set_xticks([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3])
# axs[1].set_xticklabels([r"$1 \times 10^3$", r"$2\times 10^3$", r"$3\times 10^3$", r"$4\times 10^3$", r"$5\times 10^3$", r"$6\times 10^3$", r"$7\times 10^3$", r"$8\times 10^3$", r"$9\times 10^3$", r"$10\times 10^3$"])
plt.savefig("data/TD.svg", format="svg", dpi=300)
# Plot parameters
idx = 0
_, axs = plt.subplots(4, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(b[idx][: (int(limit / update_rate))], color="C0", linewidth=0.6)
axs[0, 1].plot(
    bounds[idx][: (int(limit / update_rate)), [0, 2]], color="C0", linewidth=0.6
)
axs[1, 0].plot(f[idx][: (int(limit / update_rate))], color="C0", linewidth=0.6)
axs[1, 1].plot(
    V0[idx].squeeze()[: (int(limit / update_rate))], color="C0", linewidth=0.6
)
axs[2, 0].plot(A[idx][: (int(limit / update_rate))], color="C0", linewidth=0.6)
axs[2, 1].plot(B[idx][: (int(limit / update_rate))], color="C0", linewidth=0.6)
axs[3, 0].plot(A_cs[1][: (int(limit / update_rate))], color="C0", linewidth=0.6)
axs[3, 1].plot(A_cs[2][: (int(limit / update_rate))], color="C0", linewidth=0.6)

axs[0, 0].set_ylabel("$b_2$")
axs[0, 1].set_ylabel(r"$\underline{x}_{2, 1}, \overline{x}_{2, 1}$")
axs[1, 0].set_ylabel("$f_2$")
axs[1, 1].set_ylabel("$V_{2, 0}$")
axs[2, 0].set_ylabel("$A_2$")
axs[2, 1].set_ylabel("$B_2$")
axs[3, 0].set_ylabel("$A_{2,1}$")
axs[3, 1].set_ylabel("$A_{2,3}$")
axs[3, 1].set_xlabel(r"$\frac{t}{2}$")
axs[3, 0].set_xlabel(r"$\frac{t}{2}$")
for i in range(4):
    for j in range(2):
        axs[i, j].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
plt.savefig("data/pars.svg", format="svg")

plt.show()
