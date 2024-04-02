import pickle

import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# plt.rcParams.update({
#   "text.usetex": True,
# })

plt.rc("text", usetex=True)
plt.rc("font", size=23)
plt.style.use("bmh")

nx = 6
nx_l = 2
x_bnd = np.array([[0, -1], [1, 1]])
a_bnd = np.array([[-1], [1]])
update_rate = 2

limit = 20000

fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
files = [
    "data/academic_ex_data/line_416/distributed.pkl",
    "data/academic_ex_data/line_416/centralised.pkl",
]
count = 0
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

    axs[0].plot(
        TD[:limit],
        "o",
        color=f"C{count}",
        markersize=1,
        rasterized=True,
        label="_nolegend_",
    )
    axs[1].plot(
        R[:limit],
        "o",
        color=f"C{count}",
        markersize=1,
        rasterized=True,
        label="_nolegend_",
    )
    axs[0].set_ylabel(r"$\delta$")
    axs[1].set_ylabel(r"$L$")
    axs[1].set_xlabel(r"$t$")
    axs[0].set_ylim(-5, 15)
    axs[1].set_ylim(0, 8)
    # axs[1].set_xticks([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3])
    # axs[1].set_xticklabels([r"$1 \times 10^3$", r"$2\times 10^3$", r"$3\times 10^3$", r"$4\times 10^3$", r"$5\times 10^3$", r"$6\times 10^3$", r"$7\times 10^3$", r"$8\times 10^3$", r"$9\times 10^3$", r"$10\times 10^3$"])
    count += 1
axs[0].set_xlim(-200, 21000)
axs[1].set_xlim(-200, 21000)
axs[0].plot(30000, 30, "o", color=f"C{0}", markersize=5, rasterized=True)
axs[1].plot(30000, 30, "o", color=f"C{1}", markersize=5, rasterized=True)
fig.set_size_inches(8, 3.5)
fig.legend(["Distributed", "Centralized"], frameon=True, ncols=2)
# fig.align_ylabels()
# fig.align_xlabels()
plt.savefig("data/TD.svg", format="svg", dpi=300)

plt.show()
