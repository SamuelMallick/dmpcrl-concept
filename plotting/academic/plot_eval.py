import pickle

import matplotlib.pyplot as plt
from dmpcrl.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

with open(
    "data/example_1/evals/eval_pol.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)

_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[:, 0])
axs[1].plot(X[:, 1])
axs[2].plot(U[:, 0])

plt.show()