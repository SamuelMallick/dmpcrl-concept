from matplotlib.figure import Figure
import tikzplotlib
import matplotlib as mpl

# monkey patching to fix some issues with tikzplotlib
mpl.lines.Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
mpl.lines.Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
mpl.legend.Legend._ncol = property(lambda self: self._ncols)


def save2tikz(*figs: Figure) -> None:
    """Saves the figure to a tikz file (`.tex` extension). 
    See https://pypi.org/project/tikzplotlib/ for more details.

    Parameters
    ----------
    figs : matplotlib Figures
        One or more matplotlib figures to be converted to tikz files. These
        files will be named based on the number of the corresponding figure.
    """
    for fig in figs:
        tikzplotlib.save(
            f"figure_{fig.number}.tex",
            figure=fig,
            extra_axis_parameters={r"tick scale binop=\times"},
        )