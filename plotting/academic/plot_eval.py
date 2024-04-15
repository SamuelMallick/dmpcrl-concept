import pickle
from tikz import save2tikz
import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

time_steps = [0, 5, 10, 25, 50, 100, 500, 2500, 5000, 7500, 10000]
R_p = []
for time_step in time_steps:
    with open(
        f"data/example_1/evals/eval_pol_step_{time_step}.pkl",
        "rb",
    ) as file:
        X_p = pickle.load(file)
        U_p = pickle.load(file)
        R_p.append(sum(pickle.load(file)))

with open(
    "data/example_1/evals/nom_true_model_False.pkl",
    "rb",
) as file:
    X_nf = pickle.load(file)
    U_nf = pickle.load(file)
    R_nom = sum(pickle.load(file))

num_scen = 25
with open(
    f"data/example_1/evals/scen_{num_scen}_truemod_False.pkl",
    "rb",
) as file:
    X_sf = pickle.load(file)
    U_sf = pickle.load(file)
    R_scen_1 = sum(pickle.load(file))

with open(
    f"data/example_1/evals/scen_{num_scen}_truemod_True.pkl",
    "rb",
) as file:
    X_st = pickle.load(file)
    U_st = pickle.load(file)
    R_scen_2 = sum(pickle.load(file))

# print(
#     f"costs: policy = {sum(R_p)}, scen_false = {sum(R_sf)}, scen_true = {sum(R_st)}, nom_false = {sum(R_nf)}, nom_true = {sum(R_nt)}"
# )
# print(
#     f"viols: policy = {np.sum((X_p[:, 0]<0) | (X_p[:, 2]<0) | (X_p[:, 4]<0))}, scen_false = {np.sum((X_sf[:, 0]<0) | (X_sf[:, 2]<0) | (X_sf[:, 4]<0))}, scen_true = {np.sum((X_st[:, 0]<0) | (X_st[:, 2]<0) | (X_st[:, 4]<0))}, nom_false = {np.sum((X_nf[:, 0]<0) | (X_nf[:, 2]<0) | (X_nf[:, 4]<0))}, nom_true = {np.sum((X_nt[:, 0]<0) | (X_nt[:, 2]<0) | (X_nt[:, 4]<0))}"
# )

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
lw = 1
axs.plot([0, 20000], [R_nom] * 2, color = 'C3', linestyle='--', linewidth=lw)
axs.plot([0, 20000], [R_scen_1] * 2, color = 'C2', linestyle='--', linewidth=lw)
axs.plot([0, 20000], [R_scen_2] * 2, color = 'C1', linestyle='--', linewidth=lw)
axs.plot([2*t for t in time_steps], R_p, color='C0', linestyle='--', marker='o', linewidth=lw)
axs.set_yscale('log')
axs.legend(['nom', 'scen-1', 'scen-2', r'pol-$t$'])
axs.set_xlabel(r'$t$')
axs.set_ylabel(r'$\sum_{\tau = 0}^{100} L_\tau$')
save2tikz(plt.gcf())
plt.show()
