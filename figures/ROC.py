import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
#%%

def plot_dist(mu0 = 0.35,
              mu1=0.65,
              sigma = 0.32,
              c0ratio = 0.9,
              c1ratio = 0.9,
              textdisp = 0.025,
              tptndisp = 0,
              fp_ydisp = 0.2,
              fn_ydisp = 0.2,
              fp_fn_xdisp = 0.03,
              invert_fpfn = False,
              fontsize = 20,
              title = "Good Segmentation",
              figsize = (15,7),
              legendloc = 2,
              savename = "positive_negative.png",
              hide_fpfn = False):
    # mu0 = 0.35
    # mu1 = 0.65
    # variance = 0.10
    # variance = 0.25**2
    # sigma = math.sqrt(variance)
    variance = sigma**2
    x = np.linspace(0, 1, 1000)
    c0 = stats.norm.pdf(x, mu0, variance)
    c1 = stats.norm.pdf(x, mu1, variance)
    
    c0max = c0.max()
    
    c0 = (c0/c0max)*c0ratio
    c1= (c1/c0max)*c1ratio
    
    idx = np.argwhere(np.diff(np.sign(c0 - c1))).flatten()
    intersection = x[idx]
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    # =============================================================================
    # label positions
    # =============================================================================
    x_tn = mu0 -tptndisp
    y_tn = c0.max()*0.5
    x_tp = mu1 +tptndisp
    y_tp = c1.max()*0.5
    
    y_fp = c0[idx]*fp_ydisp
    x_fp = intersection + fp_fn_xdisp
    
    y_fn = c0[idx]*fn_ydisp
    x_fn = intersection -fp_fn_xdisp

    if invert_fpfn:
        txfp = x_fp
        txfn = x_fn

        tyfp = y_fp
        tyfn = y_fn

        x_fp = txfn
        x_fn = txfp

        y_fn = tyfp
        y_fp = tyfn

    fig = plt.figure(figsize = figsize)
    ax0 = fig.add_subplot(1,1,1)
    ax0.set_xlim([0,1])
    ax0.set_ylim([0,1])
    # ax0.get_yaxis.set_visible("False")
    ax0.set_yticks([])
    ax0.plot(x, c0,label = "negative")
    ax0.plot(x, c1,label = "positive")
    ax0.axvline(intersection, label = "threshold")
    
    ax0.text(x_tn-textdisp, y_tn, "TN", fontsize = fontsize)
    ax0.text(x_tp-textdisp, y_tp, "TP", fontsize = fontsize)
    if not hide_fpfn:
        ax0.text(x_fp-textdisp, y_fp, "FP", fontsize = fontsize)
        ax0.text(x_fn-textdisp, y_fn, "FN", fontsize = fontsize)

    ax0.set_title("Positive and Negative Classification Distributions: {}".format(title))
    ax0.legend(loc = legendloc)
    # yaxis = ax.get_yaxis()
    # yaxis.set_visible("False")
    plt.savefig(savename)


def plot_roc(mu0 = 0.35,
              mu1=0.65,
              sigma = 0.32,
              c0ratio = 0.9,
              c1ratio = 0.9,
              textdisp = 0.025,
              tptndisp = 0,
              fp_ydisp = 0.2,
              fn_ydisp = 0.2,
              fp_fn_xdisp = 0.03,
              invert_fpfn = False,
              fontsize = 20,
              title = "Good Segmentation",
              figsize = (15,7),
              legendloc = 2,
              savename = "positive_negative.png",
              hide_fpfn = False):
    # mu0 = 0.35
    # mu1 = 0.65
    # variance = 0.10
    # variance = 0.25**2
    # sigma = math.sqrt(variance)
    x = np.linspace(0, 1, 1000)

    c0 = stats.norm.pdf(x, mu0, variance)
    c1 = stats.norm.pdf(x, mu1, variance)
    
    c0max = c0.max()
    
    c0 = (c0/c0max)*c0ratio
    c1= (c1/c0max)*c1ratio
    
    idx = np.argwhere(np.diff(np.sign(c0 - c1))).flatten()
    intersection = x[idx]
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    # =============================================================================
    # label positions
    # =============================================================================
    x_tn = mu0 -tptndisp
    y_tn = c0.max()*0.5
    x_tp = mu1 +tptndisp
    y_tp = c1.max()*0.5
    
    y_fp = c0[idx]*fp_ydisp
    x_fp = intersection + fp_fn_xdisp
    
    y_fn = c0[idx]*fn_ydisp
    x_fn = intersection -fp_fn_xdisp

    if invert_fpfn:
        txfp = x_fp
        txfn = x_fn

        tyfp = y_fp
        tyfn = y_fn

        x_fp = txfn
        x_fn = txfp

        y_fn = tyfp
        y_fp = tyfn

    fig = plt.figure(figsize = figsize)
    ax0 = fig.add_subplot(1,1,1)
    ax0.set_xlim([0,1])
    ax0.set_ylim([0,1])
    # ax0.get_yaxis.set_visible("False")
    # ax0.set_yticks([])
    # ax0.plot(x, c0,label = "negative")
    # ax0.plot(x, c1,label = "positive")
    ax0.axvline(intersection, label = "threshold")
    
    ax0.text(x_tn-textdisp, y_tn, "TN", fontsize = fontsize)
    ax0.text(x_tp-textdisp, y_tp, "TP", fontsize = fontsize)
    if not hide_fpfn:
        ax0.text(x_fp-textdisp, y_fp, "FP", fontsize = fontsize)
        ax0.text(x_fn-textdisp, y_fn, "FN", fontsize = fontsize)

    ax0.set_title("Positive and Negative Classification Distributions: {}".format(title))
    ax0.legend(loc = legendloc)
    # yaxis = ax.get_yaxis()
    # yaxis.set_visible("False")
    plt.savefig(savename)

#%%
plot_dist(savename = "good_case.png")

plot_dist(mu0 = 0.25, mu1 = 0.74,
          sigma = 0.25,
          hide_fpfn = True,
          title = "Ideal Case",
          savename = "ideal_case.png")

# plot_dist(mu0 = 0.65, mu1 = 0.35,
#           hide_fpfn = False,
#           title = "Bad Case",
#           invert_fpfn = True,
#           savename = "bad_case.png")

plot_dist(mu0 = 0.75, mu1 = 0.25,
          sigma = 0.25,
          title = "Reciprocating Classes",
          hide_fpfn = True,
          savename = "reciprocating.png")

plot_dist(mu0 = 0.495, mu1 = 0.505,
          sigma = 0.35,
          title = "Worst Case",
          hide_fpfn = True,
          tptndisp = 0.05,
          savename = "worst_case.png")
#%%

x = np.linspace(0, 1, 1000)
fig = plt.figure()
ax0 = fig.add_subplot(1,1,1)
# ax0.set_xlim([0,1])
# ax0.set_ylim([0,1])
# ax0.get_yaxis.set_visible("False")
# ax0.set_yticks([])
y = np.log(x)

y = y-y.min()
ax0.plot(x, y)
# ax0.plot(x, c1,label = "positive")
# ax0.axvline(intersection, label = "threshold")

