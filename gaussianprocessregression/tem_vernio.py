import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

import pickle

#%%
def dates_to_idx(timelist):
    reference_time = min(timelist)
    t = (timelist - reference_time) / pd.Timedelta(1, "d")
    return np.asarray(t, dtype = int)
#%% importiamo i dati
df = pd.read_csv('temp_vaiano.csv',
                 skiprows = 1,
                 sep = ';',
                 names = ['data', 'max_t', 'min_t'])

# togliamo i dati non validi
#%%
#dropping nans
df.dropna(inplace = True)
#too hot
df.drop(df[df['max_t'] > 50].index, inplace = True)

df['data'] = pd.to_datetime(df['data'], dayfirst = True)

df['data_idx'] = dates_to_idx(df['data'])

df['med_t'] = (df.max_t + df.min_t)/2

df.drop(df[-1:].index, inplace = True)

df_test = df[-1000:]

plt.plot(df_test.data_idx, df_test.med_t)
#%%
#%% MISURE: DATI MANCANTI
func = lambda x: np.cos(x).flatten()

N = 100 # numero punti training
n = 200   # numero punti test

x_guess = np.linspace(df_test['data_idx'].min(), df_test['data_idx'].max(), n, dtype= int)

#%%
k2 = GPR.generate_kernel(kernel= GPR.kernel_periodic,length=15, period =365, const = 1)
k1 = GPR.generate_kernel(GPR.kernel_gaussian, length=7, const = 1)
k3 = GPR.generate_kernel(kernel=GPR.kernel_periodic, length=5,period = 15, const = 0.2)

def somma_wrap(*args, **kwargs):
    return k1(args[0], args[1]) +k2(args[0], args[1]) + k3(args[0], args[1])

create_case(df_test['data_idx'].values,x_guess, df_test['max_t'].values, kernel= somma_wrap, R = 5)
