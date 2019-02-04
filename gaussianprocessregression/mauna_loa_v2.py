
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

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
X, y = load_mauna_loa_atmospheric_co2()

ymean = y.mean()
y = y- ymean

X_ = np.linspace(X.min(), X.max()+30, 1000)


X = X[:400]
y = y[:400]

X_ = X_[:400]
plt.plot(X, y)


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
def comp_kernel2(x,y,wantgrad = False):
    
    sigma_1 = 1
    sigma_2 = 10
    sigma_3 = 0.3
    sigma_4 = 90
    sigma_5 = 1.3
    sigma_6 = 0.1
    sigma_7 = 1.2
    sigma_8 = 0.78
    sigma_9 = 0.2
    sigma_10 = 0.133
    sigma_11 = 0
    
    
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
#ok non funziona manco per il cazzo
create_case(X, X_, y, kernel = comp_kernel2, R =0.15)


#%%

create_case(
    X,
    X_,
    y,
    kernel=GPR.generate_kernel(
        GPR.kernel_gaussian, const=1, length=10),
    R = 1,
    title="Kernel Gaussiano",
    save="gaussian_maunaloa",
)
    
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
