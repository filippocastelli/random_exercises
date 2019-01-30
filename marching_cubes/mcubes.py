import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.vq as scv

from skimage import measure
from stl import mesh

import os
#%%

#NOTA:  la cartella "heatmap_lens" va piazzata nella stessa dir di esecuzione dello script

#conversione heatmap rgb in valori
def colormap2arr(arr,cmap):    
    gradient=cmap(np.linspace(0.0,1.0,100))
    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
    code,dist=scv.vq(arr2,gradient[:, :3])

    values=code.astype('float')/gradient.shape[0]

    values=values.reshape(arr.shape[0],arr.shape[1])
    values=values[::-1]
    return values
    

def threshold_array(arr, thr):
    arr[arr >= thr] = 1
    arr[arr < thr] = 0
    
    return arr

#%% carico immagini
filelist = []
list_values = []
for root, dirs, files in os.walk("./heatmap_lens"):  
    for filename in files:
        print(root + '/' + filename)
        loc  = root + '/' + filename
        filelist.append(loc)
        
        arr = plt.imread(loc)
        values = colormap2arr(arr, cm.jet)
        
        list_values.append(values)
        
#%%
list_copy = list_values.copy()
valuestack = np.array(list_copy)

#thresholding
thr_arr = threshold_array(valuestack, 0.76)


#%%

verts, faces, _, _ = measure.marching_cubes_lewiner(thr_arr, 0)

mymesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        mymesh.vectors[i][j] = verts[f[j],:]

mymesh.save('neuro.stl')