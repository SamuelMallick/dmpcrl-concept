import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from dmpcrl.mpc.mpc_admm import MpcAdmm
from model import (
    get_adj,
    get_bounds,
    get_inac_model,
    get_learnable_centralized_dynamics,
    get_model_details,
)

n, nx_l, nu_l = get_model_details()
Adj = get_adj()

# learnable parameters and their initial values for each agent
learnable_pars_init_single = {
    "V0": np.zeros((1, 1)),
    "x_lb": np.reshape([0, 0], (-1, 1)),
    "x_ub": np.reshape([1, 0], (-1, 1)),
    "b": np.zeros(nx_l),
    "f": np.zeros(nx_l + nu_l),
}


class CentralizedMpc(Mpc[cs.SX]):
    """A centralised learnable MPC controller."""

    horizon = 10
    discount_factor = 0.9

    A_l_inac, B_l_inac, A_c_l_inac = get_inac_model()

    # dictionary containing initial values for learnable parameters of centralized MPC
    learnable_pars_init = {}
    # add the parameters corresponding to each agent
    for i in range(n):
        for name, val in learnable_pars_init_single.items():
            learnable_pars_init[f"{name}_{i}"] = (
                val  # add the '_i' to indicate parameter for agent i
            )
        learnable_pars_init[f"A_{i}"] = A_l_inac
        learnable_pars_init[f"B_{i}"] = B_l_inac
        for j in range(n):
            if i != j:
                if Adj[i][j] == 1:
                    learnable_pars_init[f"A_c_{i}_{j}"] = (
                        A_c_l_inac  # add the '_i_j' to indicate coupling matrix from j to i
                    )

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor

        # init underlying optimziation problem
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # centralized state and control dimensions
        nx = n * nx_l
        nu = n * nu_l
        x_bnd_l, u_bnd_l, _ = get_bounds()
        # create bounds for global state and controls
        x_bnd = np.tile(x_bnd_l, n)
        u_bnd = np.tile(u_bnd_l, n)
        w = np.tile(
            [[1.2e2, 1.2e2]], (1, n)
        )  # penalty weight for constraint violations in cost

        # create parameters in MPC optimization scheme
        A_list = []
        B_list = []
        A_c_list = []
        V0_list = []
        x_lb_list = []
        x_ub_list = []
        b_list = []
        f_list = []
        for i in range(n):
            V0_list.append(self.parameter(f"V0_{i}", (1,)))
            x_lb_list.append(self.parameter(f"x_lb_{i}", (nx_l,)))
            x_ub_list.append(self.parameter(f"x_ub_{i}", (nx_l,)))
            b_list.append(self.parameter(f"b_{i}", (nx_l, 1)))
            f_list.append(self.parameter(f"f_{i}", (nx_l + nu_l, 1)))
            A_list.append(self.parameter(f"A_{i}", (nx_l, nx_l)))
            B_list.append(self.parameter(f"B_{i}", (nx_l, nu_l)))
            A_c_list.append([])
            for j in range(n):
                # if no coupling between i and j, A_c_list[i, j] = None, otherwise we add the parameter
                A_c_list[i].append(None)
                if i != j:
                    if Adj[i][j] == 1:
                        A_c_list[i][j] = self.parameter(
                            f"A_c_{i}_{j}",
                            (nx_l, nx_l),
                        )
        # concat the params for use in cost and constraint expressions
        V0 = cs.vcat(V0_list)
        x_lb = cs.vcat(x_lb_list)
        x_ub = cs.vcat(x_ub_list)
        b = cs.vcat(b_list)
        f = cs.vcat(f_list)

        # get centralized symbolic dynamics
        A, B = get_learnable_centralized_dynamics(
            A_list,
            B_list,
            A_c_list,
        )

        # variables (state, action, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action(
            "u", nu, lb=u_bnd[0].reshape(-1, 1), ub=u_bnd[1].reshape(-1, 1)
        )
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B @ u + b, n_in=2, n_out=1)

        # other constraints
        self.constraint("x_lb", x_bnd[0].reshape(-1, 1) + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1].reshape(-1, 1) + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            cs.sum1(V0)
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "ipopt": {
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class LocalMpc(MpcAdmm):
    """Local learnable MPC for agent in ADMM scheme."""

    horizon = 10
    discount_factor = 0.9

    A_init, B_init, A_c_l_init = get_inac_model()

    def __init__(self, num_neighbours, my_index, rho: float = 0.5) -> None:
        """Instantiate local MPC.
        My index is used to pick out own state from the grouped coupling states.
        It should be passed in via the mapping G (G[i].index(i)).
        Rho is ADMM penalty param."""

        N = self.horizon
        gamma = self.discount_factor
        self.rho = rho

        # init underlying optimziation problem
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        w = np.array(
            [[1.2e2, 1.2e2]]
        )  # penalty weight for constraint violations in cost
        x_bnd, u_bnd, _ = get_bounds()
        self.num_neighbours = num_neighbours
        self.nx_l = nx_l
        self.nu_l = nu_l

        # dictionary containing initial values for local learnable parameters
        self.learnable_pars_init = {
            "V0": np.zeros((1, 1)),
            "x_lb": np.array([0, 0]).reshape(-1, 1),
            "x_ub": np.array([1, 0]).reshape(-1, 1),
            "b": np.zeros(nx_l),
            "f": np.zeros(nx_l + nu_l),
            "A": self.A_init,
            "B": self.B_init,
        }
        for i in range(
            num_neighbours
        ):  # add a learnable param init for each neighbours coupling matrix
            self.learnable_pars_init[f"A_c_{i}"] = self.A_c_l_init

        # create parameters
        V0 = self.parameter("V0", (1,))
        x_lb = self.parameter("x_lb", (nx_l,))
        x_ub = self.parameter("x_ub", (nx_l,))
        b = self.parameter("b", (nx_l, 1))
        f = self.parameter("f", (nx_l + nu_l, 1))
        A = self.parameter("A", (nx_l, nx_l))
        B = self.parameter("B", (nx_l, nu_l))
        A_c_list: list[np.ndarray] = []  # list of coupling matrices
        for i in range(num_neighbours):
            A_c_list.append(self.parameter(f"A_c_{i}", (nx_l, nx_l)))

        # variables (state+coupling, action, slack)
        x, x_c = self.augmented_state(num_neighbours, my_index, nx_l)
        u, _ = self.action(
            "u",
            nu_l,
            lb=u_bnd[0][0],
            ub=u_bnd[1][0],
        )
        s, _, _ = self.variable("s", (nx_l, N), lb=0)

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        # dynamics - added manually due to coupling
        for k in range(N):
            coup = cs.SX.zeros(nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                f"dynam_{k}",
                A @ x[:, [k]] + B @ u[:, [k]] + coup + b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints
        self.constraint(f"x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.set_local_cost(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "ipopt": {
                "max_iter": 2000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")
