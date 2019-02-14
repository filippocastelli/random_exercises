
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data, modify_legend, predict_plot

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

X_ = np.linspace(X_full.min(), X_full.max(), 1000)
plt.figure()

plt.title("Dati concentrazione CO2 Mauna Loa")
ax = plt.gca()
dati_train,= ax.plot(X, y)
dati_test,= ax.plot(X_test, y_test)
plt.legend([dati_train, dati_test], ["dati di training", "dati di test"])
plt.xlabel("anno")
plt.ylabel("concentrazione CO2 [ppmv]")
plt.savefig("maunaloa_co2_dataoverview.png")


ymean = y_full.mean()
y_full = y_full- ymean
y_test = y_test- ymean
y = y - ymean


#%%

def comp_kernel(x,y,wantgrad = False):
    
    sigma_1 = 66
    sigma_2 = 67
    sigma_3 = 2.4
    sigma_4 = 90
    sigma_5 = 1.3
    sigma_6 = 0.66
    sigma_7 = 1.2
    sigma_8 = 0.78
    sigma_9 = 0.18
    sigma_10 = 0.133
    sigma_11 = 0.19
    
    
    #yet to implement gradient on this
    gaussian_component = GPR.generate_kernel(kernel= GPR.kernel_gaussian,
                                          length = sigma_2,
                                          const = sigma_1)
    periodic_component = GPR.generate_kernel(kernel = GPR.kernel_periodic_decay,
                                             const = sigma_3,
                                             decay = sigma_4,
                                             length = sigma_5)
    rational_quadratic_component = GPR.generate_kernel(kernel = GPR.kernel_rational_quadratic,
                                                       const = sigma_6,
                                                       length = sigma_7,
                                                       shape = sigma_8)
    noise_1 = GPR.generate_kernel(kernel = GPR.kernel_whitenoise,
                                  const = sigma_11)
    noise_2 = GPR.generate_kernel(kernel = GPR.kernel_gaussian,
                                  length = sigma_10,
                                  const = sigma_9)
    
    return gaussian_component(x,y) + periodic_component(x,y) + rational_quadratic_component(x,y) + noise_1(x,y) + noise_2(x,y)
    

#%%
sigma_1_list = np.linspace(0,60,10)
sigma_2_list = np.linspace(67,69,1)
sigma_3_list = np.linspace(0,4,10)
sigma_4_list = np.linspace(90,91,1)
sigma_5_list = np.linspace(1.3,1.5,1)
sigma_6_list = np.linspace(0,0.5,10)
sigma_7_list = np.linspace(0,60,1)
sigma_8_list = np.linspace(0,60,1)
sigma_9_list = np.linspace(0.18,1,1)
sigma_10_list = np.linspace(0.133,0.134,1)
sigma_11_list = np.linspace(0,0.1,1)

param_dict = {'sigma_1': sigma_1_list,
              'sigma_2': sigma_2_list,
              'sigma_3': sigma_3_list,
              'sigma_4': sigma_4_list,
              'sigma_5': sigma_5_list,
              'sigma_6': sigma_6_list,
              'sigma_7': sigma_7_list,
              'sigma_8': sigma_8_list,
              'sigma_9': sigma_9_list,
              'sigma_10': sigma_10_list,
              'sigma_11': sigma_11_list,
              }
optim = False
if optim == True:
    gaussian_process_regressor = GPR(X, y, kernel = GPR.kernel_mix)
    lml, params = gaussian_process_regressor.optimizer(param_dict, noiselist=False, parallel=False)
#%%    
gaus_comp2, ypreds = create_case(X, X_, y,
                            kernel = GPR.generate_kernel(GPR.kernel_mix,
                                                         sigma_1 = 66,
                                                         sigma_2 = 67,
                                                         sigma_3 = 2.4,
                                                         sigma_4 = 90,
                                                         sigma_5 = 1.3,
                                                         sigma_6 = 0.66,
                                                         sigma_7 = 1.2,
                                                         sigma_8 = 0.78,
                                                         sigma_9 = 0.18,
                                                         sigma_10 = 0.133,
                                                         sigma_11 = 0.),
                            R =0.19,
                            title = "Dati CO2 Mauna Loa",
                            load = "maunaloa_theoretically_correct_params",
                            return_regressor = True,
                            return_predictions = True,
                            draw_plot = False)
                    
plt.figure()
predict_plot(X, y, X_, ypreds, title = "GPR su dati Mauna Loa", save = "maunaloa_gpr", axlabels = ["tempo [yr]", "Concentrazione CO2 [ppmv]"] )

ax = plt.gca()
testpoints = ax.scatter(X_test, y_test, c = 'r')

#%%
x_red = X[:100]
y_red = y[:100]

X_red = np.linspace(x_red.min(), x_red.max()+30, 100)

create_case(
    x_red,
    X_red,
    y_red,
    kernel=GPR.generate_kernel(
        GPR.kernel_rational_quadratic, const=1,
        length = 1.2,
        shape = 0.8),
    R = 10,
    title="Parametri Otimizzati - Kernel Periodico, Dati Mancanti",
    save="periodico_datimancanti",
)
