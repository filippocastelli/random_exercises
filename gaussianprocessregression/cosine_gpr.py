import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

import pickle

#%% MISURE: DATI MANCANTI
func = lambda x: np.cos(x).flatten()

N = 100 # numero punti training
n = 200   # numero punti test
s = 0.1   # noise variance

period = np.pi/0.7

rng = np.random.RandomState(2)
x1 = rng.uniform(0, 3, size = (N//2,1))
x2 = rng.uniform(7, 11,size = (N//2,1))
x = np.append(x1,x2)
x.shape = (N,1)
x_guess = np.linspace(0, 11, n)
y = func(x) +s*np.random.randn(N)

#%% PLOT MISURE

plt.figure()
plt.title("Misure: dati mancanti")
ax = plt.gca()
cosine, = ax.plot(x_guess, func(x_guess))
measures = plt.scatter(x,y, c = "black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend([cosine, measures], ["f(x)", "punti training"])
plt.axvline(x=3)
plt.axvline(x=7)
plt.savefig('misure_dati_mancanti.png', bbox_inches='tight')


#%% OTTIMIZZAZIONE PARAMENTRI: KERNEL PERIODICO, DATI MANCANTI
gaus_periodic = GPR(x = x, y= y, kernel = GPR.kernel_periodic)

gaus2 = GPR(x = x,
           y = y,
           kernel = GPR.kernel_periodic)
#%%
optim = False
R_list = np.linspace(0.01, 0.2, 10)
L_list = np.linspace(0.1, 10, 50)
P_list = np.linspace(4, 9, 100)
C_list = [1]
if optim == True:
    lml, params= gaus2.optimizer(R_list, L_list, P_list,C_list)
#%% SALVO PARAMETRI
salva = False
if salva == True:
    f = open('parametri_periodico.pckl', 'wb')
    pickle.dump([lml,params], f)
    f.close()
#%% o CARCO PARAMETRI
f = open('parametri_periodico.pckl', 'rb')
#lml, params = pickle.load(f)
params = pickle.load(f)
f.close()
#%% 
#disp_params = np.round(params, 2)
#print("best parameters (log-likelihood, noise, length_scale, period, const):",np.round(lml.max(),2),disp_params[0], disp_params[1], disp_params[2], disp_params[3])
#%% PLOT REGRESSORE
create_case(x, x_guess, y,
            kernel= GPR.generate_kernel(GPR.kernel_periodic, length=params[1], period= params[2]),
            R=params[0],
            title = "Parametri Otimizzati - Kernel Periodico, Dati Mancanti",
            save = "periodico_datimancanti")

plt.axvline(x=3)
plt.axvline(x=7)
plt.savefig('periodico_datimancanti.png', bbox_inches='tight')
#%%
N_gendata = 50
x_gendata = rng.uniform(3, 7, size = (N_gendata,1)).flatten()
x_gendata.sort()
gen_data(x, x_gendata, y,
         kernel= GPR.generate_kernel(GPR.kernel_periodic, length=params[1], period= params[2]),
         R=params[0],separate_figure = False, save = "reconstructed_data",
         plot_mu = True)
#%% MISURE: DATI MANCANTI 2
func = lambda x: np.cos(x).flatten()

N = 100 # numero punti training
n = 10000   # numero punti test
s = 0.1   # noise variance

period = np.pi/0.7

rng = np.random.RandomState(2)
x1 = rng.uniform(0, 10, size = (N//4,1))
x2 = rng.uniform(15, 60,size = (3*N//4,1))
x_ext = np.append(x1,x2)
x_ext.shape = (N,1)
x_ext_guess = np.linspace(0, 60, n)
y_ext = func(x_ext) +s*np.random.randn(N)

#%% PLOT MISURE 2
plt.figure(figsize = (20, 5))
plt.title("Misure: dati mancanti 2")
ax = plt.gca()
cosine, = ax.plot(x_ext_guess, func(x_ext_guess))
measures = plt.scatter(x_ext,y_ext, c = "black")    
plt.xlabel("x")
plt.ylabel("y")
plt.legend([cosine, measures], ["f(x)", "punti training"])
plt.axvline(x=10)
plt.axvline(x=15)
plt.savefig('misure_dati_mancanti_ext.png', bbox_inches='tight')

#%% #REGRESSORE 2 - KERNEL PERIODICO
plt.figure(figsize = (20, 5))
create_case(x_ext, x_ext_guess, y_ext,
            kernel= GPR.generate_kernel(GPR.kernel_periodic, length=params[1], period= params[2]),
            R=params[0],
            title = "Parametri Otimizzati - Kernel Periodico, Dati Mancanti",
            save = "periodico_ext")

plt.axvline(x=10)
plt.axvline(x=15)
plt.savefig('periodico_ext_datimancanti.png', bbox_inches='tight')
