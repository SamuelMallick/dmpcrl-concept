import pickle

import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'Helvetica'

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
fig, axs = plt.subplots(3, 2, constrained_layout=True, sharex="col", sharey="row")
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

    # plot the results
    axs[0, count].plot(
        X[:limit, np.arange(0, nx, nx_l)], linewidth=0.25, rasterized=True
    )
    axs[1, count].plot(
        X[:limit, np.arange(1, nx, nx_l)], linewidth=0.25, rasterized=True
    )
    axs[2, count].plot(U[:limit], linewidth=0.25, rasterized=True)
    for i in range(2):
        axs[0, count].axhline(x_bnd[i][0], color="r", linewidth=2, linestyle="--")
        axs[1, count].axhline(x_bnd[i][1], color="r", linewidth=2, linestyle="--")
        axs[2, count].axhline(a_bnd[i][0], color="r", linewidth=2, linestyle="--")
    count += 1
axs[0, 0].set_ylabel("$s_1$")
axs[1, 0].set_ylabel("$s_2$")
axs[2, 0].set_ylabel("$a$")
axs[2, 0].set_xlabel(r"$t$")
axs[2, 1].set_xlabel(r"$t$")
# for ax in axs[:, 1]:
#     ax.set_xticks([])
fig.set_size_inches(8, 3.5)
# fig.align_ylabels()
# fig.align_xlabels()
plt.savefig("data/states.svg", format="svg", dpi=300)
plt.show()
