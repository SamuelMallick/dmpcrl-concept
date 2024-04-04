import pickle

import matplotlib.pyplot as plt
from dmpcrl.utils.tikz import save2tikz
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

with open(
    "data/example_1/evals/eval_pol.pkl",
    "rb",
) as file:
    X_p = pickle.load(file)
    U_p = pickle.load(file)
    R_p = pickle.load(file)

with open(
    "data/example_1/evals/nom_true_model_False.pkl",
    "rb",
) as file:
    X_nf = pickle.load(file)
    U_nf = pickle.load(file)
    R_nf = pickle.load(file)

with open(
    "data/example_1/evals/nom_true_model_True.pkl",
    "rb",
) as file:
    X_nt = pickle.load(file)
    U_nt = pickle.load(file)
    R_nt = pickle.load(file)

num_scen = 25
with open(
    f"data/example_1/evals/scen_{num_scen}_truemod_False.pkl",
    "rb",
) as file:
    X_sf = pickle.load(file)
    U_sf = pickle.load(file)
    R_sf = pickle.load(file)

with open(
    f"data/example_1/evals/scen_{num_scen}_truemod_True.pkl",
    "rb",
) as file:
    X_st = pickle.load(file)
    U_st = pickle.load(file)
    R_st = pickle.load(file)

print(
    f"costs: policy = {sum(R_p)}, scen_false = {sum(R_sf)}, scen_true = {sum(R_st)}, nom_false = {sum(R_nf)}, nom_true = {sum(R_nt)}"
)
print(
    f"viols: policy = {np.sum((X_p[:, 0]<0) | (X_p[:, 2]<0) | (X_p[:, 4]<0))}, scen_false = {np.sum((X_sf[:, 0]<0) | (X_sf[:, 2]<0) | (X_sf[:, 4]<0))}, scen_true = {np.sum((X_st[:, 0]<0) | (X_st[:, 2]<0) | (X_st[:, 4]<0))}, nom_false = {np.sum((X_nf[:, 0]<0) | (X_nf[:, 2]<0) | (X_nf[:, 4]<0))}, nom_true = {np.sum((X_nt[:, 0]<0) | (X_nt[:, 2]<0) | (X_nt[:, 4]<0))}"
)

_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
axs[0].plot(X_p[:, 0])
axs[1].plot(X_p[:, 1])
axs[2].plot(U_p[:, 0])
axs[3].plot(np.cumsum(R_p))

# axs[0].plot(X_nf[:, 0])
# axs[1].plot(X_nf[:, 1])
# axs[2].plot(U_nf[:, 0])
# axs[3].plot(np.cumsum(R_nf))

# axs[0].plot(X_nt[:, 0])
# axs[1].plot(X_nt[:, 1])
# axs[2].plot(U_nt[:, 0])
# axs[3].plot(np.cumsum(R_nt))

axs[0].plot(X_sf[:, 0])
axs[1].plot(X_sf[:, 1])
axs[2].plot(U_sf[:, 0])
axs[3].plot(np.cumsum(R_sf))

axs[0].plot(X_st[:, 0])
axs[1].plot(X_st[:, 1])
axs[2].plot(U_st[:, 0])
axs[3].plot(np.cumsum(R_st))

plt.show()
