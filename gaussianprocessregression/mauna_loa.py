import pandas as pd
import numpy as np

import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
sns.set(color_codes = True)

from GP import GPR
from GP import f1, create_case, prior, post, clr, gen_data

import pickle

#%%

data_monthly = pd.read_csv(pm.get_data("monthly_in_situ_co2_mlo.csv"), header=56)

# - replace -99.99 with NaN
data_monthly.replace(to_replace=-99.99, value=np.nan, inplace=True)

# fix column names
cols = ["year", "month", "--", "--", "CO2", "seasonaly_adjusted", "fit",
        "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]
data_monthly.columns = cols
cols.remove("--"); cols.remove("--")
data_monthly = data_monthly[cols]

# drop rows with nan
data_monthly.dropna(inplace=True)

# fix time index
data_monthly["day"] = 15
data_monthly.index = pd.to_datetime(data_monthly[["year", "month", "day"]])
cols.remove("year"); cols.remove("month")
data_monthly = data_monthly[cols]

data_monthly.head(5)

#%%

# function to convert datetimes to numbers that are useful to algorithms
#   this will be useful later when doing prediction

def dates_to_idx(timelist):
    reference_time = pd.to_datetime('1958-03-15')
    t = (timelist - reference_time) / pd.Timedelta(1, "Y")
    return np.asarray(t)

t = dates_to_idx(data_monthly.index)

# normalize CO2 levels
y = data_monthly["CO2"].values
first_co2 = y[0]
std_co2 = np.std(y)
y_n = (y - first_co2) / std_co2

data_monthly = data_monthly.assign(t = t)
data_monthly = data_monthly.assign(y_n = y_n)
#%%


# split into training and test set
sep_idx = data_monthly.index.searchsorted(pd.to_datetime("2003-12-15"))
data_early = data_monthly.iloc[:sep_idx+1, :]
data_later = data_monthly.iloc[sep_idx:, :]

#%%
plt.figure()
ax = plt.gca()
training_data_plot, = ax.plot(data_early['t'], data_early['y_n'], c = 'black')

test_data_plot, = ax.plot(data_later['t'], data_later['y_n'], c = 'red')

plt.title("Mauna Loa C02")
plt.legend([training_data_plot, test_data_plot], ["train", "test"])


#%%

#using arrays makes me save time in pandas access

t_early = data_early['t'].values
y_early = data_early['y_n'].values

t_later = data_later['t'].values
y_later = data_later['y_n'].values


#%% proposed kernel

def comp_kernel(x,y,
                sigma_1 = 1,
                sigma_2 = 1,
                sigma_3 = 1,
                sigma_4 = 1,
                sigma_5 = 1,
                sigma_6 = 1,
                sigma_7 = 1,
                sigma_8 = 1,
                sigma_9 = 1,
                sigma_10 = 1,
                sigma_11 = 1,
                wantgrad = False):
    
    
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
    
    
    

create_case(x,x_guess, y, kernel= comp_kernel, R = 0.1)


#%%
