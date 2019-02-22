from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

from GP_REBOOT import GPR_reboot

#%%

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


#%%
X_full, y_full = load_mauna_loa_atmospheric_co2()
cutoff = 100

X = X_full[:-cutoff]
y = y_full[:-cutoff]

X_test = X_full[-cutoff :]
y_test = y_full[-cutoff :]

X_ = np.linspace(X_full.min(), X_full.max() + 20, 1000)
plt.figure()

plt.title("Dati concentrazione CO2 Mauna Loa")
ax = plt.gca()
dati_train,= ax.plot(X, y)
dati_test,= ax.plot(X_test, y_test)
plt.legend([dati_train, dati_test], ["dati di training", "dati di test"])
plt.xlabel("anno")
plt.ylabel("concentrazione CO2 [ppmv]")
plt.savefig("maunaloa_co2_dataoverview.png")


#ymean = y_full.mean()
#y_full = y_full- ymean
#y_test = y_test- ymean
#y = y - ymean

#%%
salta = True
if salta != True:
    # Kernel with parameters given in GPML book
    k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
    k2 = 2.4**2 * RBF(length_scale=90.0) \
        * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
    # medium term irregularity
    k3 = 0.66**2 \
        * RationalQuadratic(length_scale=1.2, alpha=0.78)
    k4 = 0.18**2 * RBF(length_scale=0.134) \
        + WhiteKernel(noise_level=0.19**2)  # noise terms
    kernel_gpml = k1 + k2 + k3 + k4
    
    gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                                  optimizer=None, normalize_y=True)
    gp.fit(X, y)
    
    print("GPML kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f"
          % gp.log_marginal_likelihood(gp.kernel_.theta))
    
    # Kernel with optimized parameters
    k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k2 = 2.0**2 * RBF(length_scale=100.0) \
        * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                         periodicity_bounds="fixed")  # seasonal component
    # medium term irregularities
    k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    k4 = 0.1**2 * RBF(length_scale=0.1) \
        + WhiteKernel(noise_level=0.1**2,
                      noise_level_bounds=(1e-3, np.inf))  # noise terms
    kernel = k1 + k2 + k3 + k4
    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                  normalize_y=True,
                                  n_restarts_optimizer = 0)
    gp.fit(X, y)
    
    print("\nLearned kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f"
          % gp.log_marginal_likelihood(gp.kernel_.theta))
    
    X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    
    # Illustration
    plt.scatter(X, y, c='k')
    plt.plot(X_, y_pred)
    plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                     alpha=0.5, color='k')
    plt.xlim(X_.min(), X_.max())
    plt.xlabel("Year")
    plt.ylabel(r"CO$_2$ in ppm")
    plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
    plt.tight_layout()
    plt.show()


#%%
model_params = {
'RBF_const': 67,
'RBF_length':90,
'RBFperiodic_const':2.4,
'RBFperiodic_length':100,
'PERIODIC_length':1.3,
'RADQUAD_const':0.66, 
'RADQUAD_length':1.2,
'RADQUAD_shape':0.78,
'RBFnoise_length':1.33,
'RBFnoise_const':0.18}

gpr = GPR_reboot(x = X,
                 y = y,
                 x_guess = X_,
                 kernel = GPR_reboot.mauna_loa_example_kernel2,
                 kernel_params = model_params,
                 normalize_y = True,
                 R = 0.19)


gpr.predict()

#%%
gpr.plot(title = "Gaussian Process Regression, dati Mauna Loa",
         axlabels = ["anno [yr]", "concentrazione CO2 [ppmv]"],
         save = "mauna_loa_regression",
         return_ax = True,
         figsize = [20,10])
#%%
ax = gpr.plot(title = "Gaussian Process Regression, Dettaglio",
         axlabels = ["anno [yr]", "concentrazione CO2 [ppmv]"],
         save = "mauna_loa_prediction_detail",
         return_ax = True,
         figsize = [20,10])
ax.scatter(X_test, y_test)