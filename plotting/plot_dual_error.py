import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")
plt.rc("font", size=14)
plt.style.use("bmh")
from dmpcrl.utils.tikz import save2tikz

iters = [1, 2, 5, 10, 20, 30, 40, 50, 70, 100]
errors = [
    0.32175163756730835,
    0.010231478997510639,
    0.00010059943956012176,
    5.315544517759367e-07,
    4.93572939251872e-07,
    4.935729394433562e-07,
]
errors_2 = [
    0.42900035936186515,
    0.059175278170436954,
    0.013035984010554365,
    0.00014196658974968676,
    8.177999094424456e-08,
    8.241436840077202e-11,
    7.205275879153803e-13,
    6.977461171070973e-13,
    6.974329424736735e-13,
    6.979407316638065e-13,
]
plt.figure(figsize=(10, 4))
plt.semilogy(iters[:], errors_2[:], "-o")
plt.xlabel(r"$\tau$")
plt.ylabel(
    r"$\sum_{i=0}^3 \| \bm{\lambda}_i^\star - \bm{\lambda}_i^{\star, \tau} \| + \| \bm{\mu}_i^\star - \bm{\mu}_i^{\star, \tau}\|$"
)
save2tikz(plt.gcf())
plt.show()
