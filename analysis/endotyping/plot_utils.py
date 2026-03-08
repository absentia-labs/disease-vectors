import numpy as np, pandas as pd, matplotlib.pyplot as plt

fontsize=12
def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None,
    colors=None,
    max_death=20
):
    ax = ax or plt.gca()
    plt.style.use(colormap)
    xlabel, ylabel = "Birth", "Death"
    if not isinstance(diagrams, list):
        diagrams = [diagrams]
    if labels is None:
        labels = ["$H_{{{}}}$".format(i) for i, _ in enumerate(diagrams)]
    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]
    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]
    if not xy_range:
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min
        buffer = 1 if xy_range == 0 else x_r / 5
        x_down, x_up = ax_min - buffer / 2, ax_max + buffer
        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range
    yr = y_up - y_down
    if lifetime:
        diagonal = False
        y_down = -yr * 0.05
        y_up = y_down + yr
        ylabel = "Lifetime"
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]
        ax.plot([x_down, x_up], [0, 0], c=ax_color)
    if diagonal:
        #ax.plot([x_down, x_up], [x_down, x_up], ls="dotted", c="gray", lw=1.5, zorder=-10)
        ax.plot([-2, max_death+2], [-2, max_death+2], ls="dotted", c="gray", lw=1.5, zorder=-10)
    if has_inf:
        #b_inf = y_down + yr * 0.95
        b_inf = max_death 
        ax.plot([-2, max_death+2], [b_inf, b_inf], "--", c="k", label=r"$\infty$", lw=1.5)
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf
    for dgm, label, c in zip(diagrams, labels, colors):
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, color=c, alpha=0.25, linewidths=1, edgecolors=c)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.set_xlim(-2, max_death+2)
    ax.set_ylim(-2, max_death+2)
    ticks = list(np.arange(0, max_death, 5)) + [max_death]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_aspect("equal", "box")
    for spine in ax.spines.values(): spine.set_linewidth(0)


    if legend:
        ax.legend(loc="lower right")
    ax.legend(fontsize=fontsize)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(fontsize)
        #tick_label.set_text(str(int(float(tick_label.get_text()))))

    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    [ax.spines[s].set_linewidth(0) for s in ['right','top']]
    [ax.spines[s].set_linewidth(2) for s in ['bottom','left']]


import io
class Barcode:
    def __init__(self, diagrams, verbose=False):

        if not isinstance(diagrams, list):
            diagrams = [diagrams]

        self.diagrams = diagrams
        self._verbose = verbose
        self._dim = len(diagrams)

    def plot_barcode(self, figsize=None, show=True, export_png=False, dpi=100, **kwargs):
        if figsize is None:
            figsize = (6, 4)

        return self._plot_Hn(
            figsize=figsize,
            show=show,
            export_png=export_png,
            dpi=dpi,
            **kwargs
        )

    def _plot_Hn(self, *, figsize, show, export_png, dpi, **kwargs):
        out = []

        for dim, diagram in enumerate(self.diagrams):
            if dim > 0:continue
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

            self._plot_many_bars(dim, diagram, 0, [ax], **kwargs)
        return ax


    def _plot_many_bars(self, dim, diagram, idx, ax, **kwargs):
        number_of_bars = len(diagram)
        if self._verbose:
            print("Number of bars in dimension %d: %d" % (dim, number_of_bars))

        if number_of_bars > 0:
            births = np.vstack([(elem[0], i) for i, elem in enumerate(diagram)])
            deaths = np.vstack([(elem[1], i) for i, elem in enumerate(diagram)])

            inf_bars = np.where(np.isinf(deaths))[0]
            max_death = max(deaths[np.isfinite(deaths[:, 0]), 0].max(), kwargs["max_death"])

            number_of_bars_fin = births.shape[0] - inf_bars.shape[0]
            number_of_bars_inf = inf_bars.shape[0]
            del kwargs["max_death"]
            _ = [self._plot_a_bar(ax[idx], birth, deaths[i], max_death, **kwargs) for i, birth in enumerate(births)]

        # the line below is to plot a vertical red line showing the maximal finite bar length
        ax[idx].plot(
            [max_death, max_death],
            [0, number_of_bars - 1],
            c='r',
            linestyle='--',
            linewidth=1.5
        )

        #title = "H%d barcode: %d finite, %d infinite" % (dim, number_of_bars_fin, number_of_bars_inf)
        #ax[idx].set_title(title, fontsize=9)
        ax[idx].set_yticks([])

        for loc in ('right', 'left', 'top'):
            ax[idx].spines[loc].set_visible(False)
        ax[idx].spines['bottom'].set_linewidth(2)

    @staticmethod
    def _plot_a_bar(ax, birth, death, max_death, c='k', linestyle='-', linewidth=0.5):
        if np.isinf(death[0]):
            death[0] = 1.05 * max_death
            c='k'
            linewidth=1.5
            ax.plot(death[0],death[1],c=c,markersize=6,marker='>',)
        ax.plot( [birth[0], death[0]], [birth[1], death[1]],  c=c, alpha=0.8, linestyle=linestyle, linewidth=linewidth)