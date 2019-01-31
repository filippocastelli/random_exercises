import theano.tensor as tt
import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#%%

def gen_bivariate_data(means, stds, corr, n_obs = 1000):
    
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr,           stds[1]**2]] 
    
    m = np.random.multivariate_normal(means, covs, n_obs).T
    
    return m

def binarize(data):
    data[data <= 0.5] = 0
    data[data > 0.5] = 1
    
    return data

#%%
SEED = 326402
n_pazienti = 20

np.random.seed(SEED)


#%%

genes_01 = gen_bivariate_data(means = [0.5, 0.5],
                       stds = [0.5, 0.5],
                       corr = 0.99,
                       n_obs = n_pazienti)

genes_01_bin = (binarize(genes_01.copy())).astype(int)


plt.figure()
plt.title("geni 0,1")
plt.scatter(genes_01[0], genes_01[1])

plt.scatter(genes_01_bin[0], genes_01_bin[1])


plt.figure()
plt.title("geni 2,3")
genes_23 = gen_bivariate_data(means = [0.5, 0.5],
                       stds = [0.5, 0.5],
                       corr = -0.99,
                       n_obs = n_pazienti)

genes_23_bin = (binarize(genes_23.copy())).astype(int)

plt.scatter(genes_23[0], genes_23[1])

plt.scatter(genes_23_bin[0], genes_23_bin[1])

#%%

def namelist(n):
    
    nlist = []
    for i in range(n):
        string = "paziente" + str(i)
        
        nlist.append(string)
        
    return nlist

    
dati = pd.DataFrame(index = namelist(n_pazienti),
                    data = {'gene0': genes_01_bin[0],
                            'gene1': genes_01_bin[1],
                            'gene2': genes_23_bin[0],
                            'gene4': genes_23_bin[1]})


dati = dati.transpose()
        
        
    
dati_old = pd.DataFrame(index = ['gene0','gene1', 'gene2', 'gene3','gene4'],
                    data = {'paziente1':    [0,1,0,0,0],
                            'paziente2':    [0,1,0,0,0],
                            'paziente3':    [1,0,1,1,0],                                        
                            'paziente4':    [1,0,1,1,0],
                            'paziente5':    [0,1,0,0,0],
                            'paziente6':    [0,1,0,0,0],
                            'paziente7':    [0,0,0,0,0],
                            'paziente8':    [0,1,1,1,0],
                            'paziente9':    [0,1,0,0,0],
                            'paziente11':   [0,1,1,1,0],
                            'paziente12':   [0,1,1,1,0],
                            'paziente13':   [1,0,0,0,0],
                            'paziente14':   [1,0,0,0,0],
                            'paziente15':   [1,0,0,0,0],
                            'paziente16':   [1,0,0,0,0],
                            'paziente17':   [0,1,1,1,0],
                            'paziente18':   [0,1,1,1,0],
                            'paziente19':   [0,1,0,0,0],
                            'paziente20':   [0,1,0,0,0],
                            'paziente21':   [0,1,1,1,0],
                            'paziente22':   [1,0,1,1,0],
                            'paziente23':   [1,0,0,0,0],
                            'paziente24':   [1,0,0,0,0],
                            
                            
                            }
                    )




#maybe bad label format?
#dati.replace(0, -1, inplace = True)
#%%

dati_t = dati.transpose()

plt.matshow(dati_t.cov())
plt.title('correlation plot')

plt.matshow(dati_t.corr())
plt.title('covariance plot')

#%%
    
model = pm.Model()

    
SEED = 12345

with model:
    packed_L = pm.LKJCholeskyCov('Packed_L', n=len(dati), eta = 2, sd_dist = pm.HalfCauchy.dist(1.5))
    L = pm.expand_packed_triangular(len(dati), packed_L)
#    sigma = pm.Wishart("Sigma", nu = 1, V = sigma0, shape = (3,3))
    sigma = pm.Deterministic('Sigma', L.dot(L.T))
#    mu = pm.Normal('mu', 0., 10., shape=3, testval = dati.mean(axis=1))
    logits = pm.MvNormal("Logits", mu = 0*np.ones(len(dati)), cov = sigma, shape = len(dati))
    p = pm.Deterministic('p', tt.exp(logits)/(1 + tt.exp(logits)))
    observed = pm.Binomial("Observed", p = p, n = 1, observed = dati.values.T)
    trace  = pm.sample(random_seed = SEED, cores = 1)
    
    
    
#%%

sigmas = trace['Sigma']

plt.matshow(sigmas.mean(axis = 0))
plt.matshow(sigmas.std(axis = 0))


corrs = np.zeros(shape= sigmas.shape)

for i, sigma in enumerate(sigmas):
#    diag = np.power(np.diag(np.diag(sigma), -0.5)
    
    diag = np.diag(np.power(np.diag(sigma), -0.5))
    
    corrs[i] = np.dot(diag, np.dot(sigma, diag))
    
    
    
plt.matshow(corrs[10])
plt.matshow(corrs.mean(axis=0))
#plt.matshow(corrs.std(axis = 0))
