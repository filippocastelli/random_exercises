import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

#%%


def plot_dist(
    mu0=0.35,
    mu1=0.65,
    sigma=0.32,
    c0ratio=0.9,
    c1ratio=0.9,
    textdisp=0.025,
    tptndisp=0,
    fp_ydisp=0.2,
    fn_ydisp=0.2,
    fp_fn_xdisp=0.03,
    invert_fpfn=False,
    fontsize=20,
    title="Good Segmentation",
    figsize=(15, 7),
    legendloc=2,
    savename= None,
    hide_fpfn=False,
    hide_tptn=False,
    out_fig=False,
):
    # mu0 = 0.35
    # mu1 = 0.65
    # variance = 0.10
    # variance = 0.25**2
    # sigma = math.sqrt(variance)
    variance = sigma ** 2
    x = np.linspace(0, 1, 1000)
    c0 = stats.norm.pdf(x, mu0, variance)
    c1 = stats.norm.pdf(x, mu1, variance)

    c0max = c0.max()

    c0 = (c0 / c0max) * c0ratio
    c1 = (c1 / c0max) * c1ratio

    idx = np.argwhere(np.diff(np.sign(c0 - c1))).flatten()
    intersection = x[idx]
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    # =============================================================================
    # label positions
    # =============================================================================
    x_tn = mu0 - tptndisp
    y_tn = c0.max() * 0.5
    x_tp = mu1 + tptndisp
    y_tp = c1.max() * 0.5

    y_fp = c0[idx] * fp_ydisp
    x_fp = intersection + fp_fn_xdisp

    y_fn = c0[idx] * fn_ydisp
    x_fn = intersection - fp_fn_xdisp

    if invert_fpfn:
        txfp = x_fp
        txfn = x_fn

        tyfp = y_fp
        tyfn = y_fn

        x_fp = txfn
        x_fn = txfp

        y_fn = tyfp
        y_fp = tyfn

    # =============================================================================
    # actual plot
    # =============================================================================
    fig = plt.figure(figsize=figsize)
    cols = 2 if out_fig else 1
    ax0 = fig.add_subplot(1, cols, 1)
    ax0.set_xlim([0, 1])
    ax0.set_ylim([0, 1])
    # ax0.get_yaxis.set_visible("False")
    ax0.set_yticks([])
    ax0.plot(x, c0, label="negative")
    ax0.plot(x, c1, label="positive")
    ax0.axvline(intersection, label="threshold")

    # =============================================================================
    # overlays
    # =============================================================================
    if not hide_tptn:
        ax0.text(x_tn - textdisp, y_tn, "TN", fontsize=fontsize)
        ax0.text(x_tp - textdisp, y_tp, "TP", fontsize=fontsize)
    if not hide_fpfn:
        ax0.text(x_fp - textdisp, y_fp, "FP", fontsize=fontsize)
        ax0.text(x_fn - textdisp, y_fn, "FN", fontsize=fontsize)

    ax0.set_title(
        "Positive and Negative Classification Distributions: {}".format(title)
    )
    ax0.legend(loc=legendloc)
    # yaxis = ax.get_yaxis()
    # yaxis.set_visible("False")
    if savename is not None:
        plt.savefig(savename)

    if out_fig == True:
        return fig


def plot_roc(
    case="good",
    figsize=(7, 7),
    title="Good Classifier",
    savename="roc_good",
    param=0.1,
    legendloc=2,
    figure=None,
    linewidth=2,
):

    x = np.linspace(0.00001, 1, 1000)

    if case == "good":
        # y = (1- (1-x)**2) + (1- np.sqrt(x))
        y = 1 - (1 - x ** (1 - param)) * (1 - np.sqrt(x))
        y = y - y.min()
        y = y / y.max()
        # y = (np.log(x) -1)
        # y = (y - y.min())/(-y.min())

    elif case == "linear":
        y = x
    elif case == "ideal":
        y = np.ones_like(x)
        y[0] = 0
    elif case == "reciprocating":
        y = np.zeros_like(x)
        y[-1] = 1

    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
    ax0 = fig.add_subplot(1, 2, 2)
    ax0.set_xlim([-0.01, 1])
    ax0.set_ylim([0, 1.01])
    ax0.plot(x, y, label="ROC", linewidth=linewidth)
    if case != "linear":
        ax0.plot(x, x, "--", label="indecisive classifier ROC")

    ax0.legend(loc=legendloc)
    ax0.set_title("ROC Curve: {}".format(title))
    plt.savefig(savename)


#%%

figsize = (20, 5)

title = "Good Classifier"
fig_good = plot_dist(
    title="Good Classifier", out_fig=True, figsize=figsize
)

plot_roc(figure=fig_good, savename="roc_good")

title = "Ideal Classifier"
fig_ideal = plot_dist(
    mu0=0.25,
    mu1=0.74,
    sigma=0.25,
    hide_fpfn=True,
    title=title,
    out_fig=True,
    figsize=figsize,
)

plot_roc(case="ideal", figure=fig_ideal, title=title, savename="roc_ideal", legendloc=4)


title = "Reciprocating Classes"

fig_reciprocating = plot_dist(
    mu0=0.75,
    mu1=0.25,
    sigma=0.25,
    figsize=figsize,
    title="Reciprocating Classes",
    hide_fpfn=True,
    out_fig=True,
)

plot_roc(
    figure=fig_reciprocating,
    case="reciprocating",
    title=title,
    savename="roc_reciprocating",
)

title = "Indecisive Classifier"

fig_indecisive = plot_dist(
    title = title,
    mu0=0.495,
    mu1=0.505,
    sigma=0.32,
    hide_tptn=True,
    hide_fpfn=True,
    out_fig=True,
    figsize=figsize,
)

plot_roc(figure=fig_indecisive, case="linear", title=title, savename="roc_indecisive")
