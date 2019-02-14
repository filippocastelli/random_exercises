import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

import pickle

#%% MISURE: DATI MANCANTI
func = lambda x: np.cos(x).flatten()

N = 100  # numero punti training
n = 200  # numero punti test
s = 0.1  # noise variance

period = np.pi / 0.7

rng = np.random.RandomState(2)
x1 = rng.uniform(0, 3, size=(N // 2, 1))
x2 = rng.uniform(7, 11, size=(N // 2, 1))
x = np.append(x1, x2)
x.shape = (N, 1)
x_guess = np.linspace(0, 11, n)
y = func(x) + s * np.random.randn(N)
#%%


plt.figure()
plt.title("Misure: dati mancanti")
ax = plt.gca()
cosine, = ax.plot(x_guess, func(x_guess))
measures = plt.scatter(x, y, c="black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend([cosine, measures], ["f(x)", "punti training"])
plt.axvline(x=3)
plt.axvline(x=7)
# plt.savefig('misure_dati_mancanti.png', bbox_inches='tight')

#%% OTTIMIZZAZIONE PARAMENTRI: KERNEL PERIODICO, DATI MANCANTI
gaus_periodic = GPR(x=x, y=y, kernel=GPR.kernel_periodic)

gaus2 = GPR(x=x, y=y, kernel=GPR.kernel_periodic)

#%%
optim = True
R_list = np.linspace(0.01, 0.2, 10)
L_list = np.linspace(0.1, 10, 10)
P_list = np.linspace(4, 9, 10)
C_list = [1]

param_dict = {"const": C_list, "period": P_list, "length": L_list}
if optim == True:
#    lml, pms = gaus2.optimizer(param_dict)
    
    lml, optim_theta = gaus2.optimizer(param_dict, noiselist = R_list, parallel = False)

#%%
optim = True
# R_list = np.linspace(0.01, 0.2, 10)
L_list = np.linspace(0.1, 10, 50)
P_list = np.linspace(4, 9, 100)
C_list = [1]

theta0 = {"const": 1.0, "period": 5.0, "length": 5.0, 'noise': 0.001}
if optim == True:
    minimizer, optim_theta = gaus2.grad_optimizer(theta0)


#%% PLOT REGRESSORE
create_case(
    x,
    x_guess,
    y,
    kernel=GPR.generate_kernel(
        GPR.kernel_periodic, const=optim_theta['const'], period=optim_theta['period'], length=optim_theta['length']
    ),
    R=optim_theta['noise'],
    title="Parametri Otimizzati - Kernel Periodico, Dati Mancanti",
    save="periodico_datimancanti",
)

plt.axvline(x=3)
plt.axvline(x=7)
plt.savefig("periodico_datimancanti.png", bbox_inches="tight")
