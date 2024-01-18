import pickle

import matplotlib.pyplot as plt
import numpy as np
from dmpcrl.utils.tikz import save2tikz
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
from matplotlib.ticker import FormatStrFormatter

# plt.rcParams.update({
#   "text.usetex": True,
# })

plt.rc("text", usetex=True)
plt.rc("font", size=18)
plt.style.use("bmh")

nx = 6
nx_l = 2
x_bnd = np.array([[0, -1], [1, 1]])
a_bnd = np.array([[-1], [1]])
update_rate = 2

limit = 20000
lw=1
x_data = [i for i in range(0, limit, 2)]

files = ["data/academic_ex_data/line_416/distributed.pkl", "data/academic_ex_data/line_416/centralised.pkl"]
fig, axs = plt.subplots(4, 2, constrained_layout=True, sharex=True)
col = 0
for filename in files:
    with open(
        filename,
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
    # Plot parameters
    idx = 0
    axs[0, 0].plot(x_data, b[idx][: (int(limit / update_rate))], color=f"C{col}", linewidth=lw)
    axs[0, 1].plot(x_data,
        bounds[idx][: (int(limit / update_rate)), [0, 2]], color=f"C{col}", linewidth=lw
    )
    axs[1, 0].plot(x_data, f[idx][: (int(limit / update_rate))], color=f"C{col}", linewidth=lw)
    axs[1, 1].plot(x_data,
        V0[idx].squeeze()[: (int(limit / update_rate))], color=f"C{col}", linewidth=lw
    )
    axs[2, 0].plot(x_data, A[idx][: (int(limit / update_rate))], color=f"C{col}", linewidth=lw)
    axs[2, 1].plot(x_data, B[idx][: (int(limit / update_rate))], color=f"C{col}", linewidth=lw)
    axs[3, 0].plot(x_data, A_cs[1][: (int(limit / update_rate))], color=f"C{col}", linewidth=lw)
    axs[3, 1].plot(x_data, A_cs[2][: (int(limit / update_rate))], color=f"C{col}", linewidth=lw)

    col += 1

axs[0, 0].set_ylabel("$b_2$")
axs[0, 1].set_ylabel(r"$\underline{x}_{2, 1}, \overline{x}_{2, 1}$")
axs[1, 0].set_ylabel("$f_2$")
axs[1, 1].set_ylabel("$V_{2, 0}$")
axs[2, 0].set_ylabel("$A_2$")
axs[2, 1].set_ylabel("$B_2$")
axs[3, 0].set_ylabel("$A_{2,1}$")
axs[3, 1].set_ylabel("$A_{2,3}$")
axs[3, 1].set_xlabel(r"$t$")
axs[3, 0].set_xlabel(r"$t$")
for i in range(4):
    for j in range(2):
        axs[i, j].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#axs[3, 0].legend(['Distributed', 'Centralized'])
fig.set_size_inches(7, 4)
#fig.align_ylabels()
#fig.align_xlabels()
plt.savefig("data/pars_overlay.svg", format="svg")
plt.show()
