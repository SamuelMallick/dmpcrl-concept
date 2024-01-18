import matplotlib.pyplot as plt
from model_Hycon2 import get_model_details, get_P_tie

(
    n,
    nx_l,
    nu_l,
    Adj,
    ts,
    prediction_length,
    discount_factor,
    u_lim,
    theta_lim,
    w,
    load_noise_bnd,
) = get_model_details()


def plot_power_system_data(TD, R, TD_eps, R_eps, X, U, param_dict=None):
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(TD, "o", markersize=1)
    axs[1].plot(R, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(TD_eps, "o", markersize=1)
    axs[1].semilogy(R_eps, "o", markersize=1)

    _, axs = plt.subplots(6, 1, constrained_layout=True, sharex=True)
    P_tie = get_P_tie()
    for i in range(n):
        axs[0].plot(X[:, i * nx_l])
        axs[0].axhline(theta_lim, color="r")
        axs[0].axhline(-theta_lim, color="r")
        axs[1].plot(X[:, i * (nx_l) + 1])
        axs[2].plot(X[:, i * (nx_l) + 2])
        axs[3].plot(X[:, i * (nx_l) + 3])
        for j in range(n):
            if P_tie[i, j] != 0:
                axs[5].plot(P_tie[i, j] * (X[:, i * nx_l] - X[:, j * nx_l]))
    axs[4].plot(U)

    if param_dict is not None:
        _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
        for name in param_dict:
            if len(param_dict[name].shape) <= 2:  # TODO dont skip plotting Q
                axs.plot(param_dict[name].squeeze())

    plt.show()
