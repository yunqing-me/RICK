import math
import random
import functools
import operator
from turtle import pen

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


from model import Discriminator
from torchvision import transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader
import os, PIL
from fid import calc_fid
import numpy as np
import pickle
from scipy import linalg


import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull

def load_embeddings(name, type):
    embeddings = torch.load('embeddings_{}/{}.pt'.format(type, name))
    print("Loaded {} embeddings with size = {} ".format(name, embeddings.size()))
    return embeddings


def perform_pca(source_embeddings, n_components=128):
    pca = PCA(n_components=n_components)
    pca.fit(source_embeddings)
    return pca


def perform_tnse(embeddings):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30, n_iter=2000)
    tsne_fts = tsne.fit_transform(embeddings)
    print(tsne.n_iter_)
    return tsne_fts



def get_tsne_results(list):
    all_pca_fts = None
    all_labels = []

    source_embeddings = load_embeddings('ffhq', 'lpips')
    pca = PCA(n_components=64, whiten=True)
    
    for idx, l in enumerate(list):
        target_embeddings = load_embeddings(l, 'lpips')

        if idx == 0:
            all_pca_fts = source_embeddings
            all_pca_fts = np.concatenate((all_pca_fts, target_embeddings), axis=0)
            
            labels = [0]*source_embeddings.shape[0] + [idx+1]* target_embeddings.shape[0]
            all_labels.extend(labels)

        else:
            all_pca_fts = np.concatenate((all_pca_fts, target_embeddings), axis=0)
            labels = [idx+1]* target_embeddings.shape[0]
            all_labels.extend(labels)


    # Perform PCA
    print(all_pca_fts.shape)
    pca.fit(all_pca_fts[:70000, :])
    all_pca_fts = pca.transform(all_pca_fts)
    print(all_pca_fts.shape)

    return all_pca_fts, np.asarray(all_labels)



datasets = ['babies', 
            'sunglasses', 
            'metfaces', 
        
            'afhq_cat', 
            'afhq_dog', 
            'afhq_wild', 
            ]

all_pca_fts, labels = get_tsne_results(datasets)

# Perform tsne
tsne_fts = perform_tnse(all_pca_fts)
print(tsne_fts.shape)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.xmargin'] = 0
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(rasterized=True)
ax.grid(True)

datasets = ['ffhq',
            'babies', 
            'sunglasses', 
            'metfaces', 

            'afhq_cat',  
            'afhq_dog', 
            'afhq_wild', 
            ]

cdict = {
    'ffhq':'#457b9d', 

    'babies' : '#606c38', 
    'metfaces' : '#b5838d', 
    'sunglasses' : '#80ffe8', 

    'afhq_cat' : '#e63946', 
    'afhq_dog' : '#b5179e', 
    'afhq_wild' : '#fca311', 

}


label_names = {
    'ffhq':'FFHQ', 

    'babies' : 'Babies', 
    'metfaces' : 'MetFaces', 
    'sunglasses' : 'Sunglasses', 

    'afhq_cat' : 'Cat', 
    'afhq_dog' : 'Dog', 
    'afhq_wild' : 'Wild', 

}

for g in np.unique(labels):
    ix = np.where(labels == g)
    
    color = cdict[ datasets[g] ]
    label = label_names[datasets[g]]

    centroid = np.mean(tsne_fts[ix, :], axis=1).flatten()
    ax.scatter(tsne_fts[ix, 0], tsne_fts[ix, 1], alpha = 0.2, s=10, color=color)
    ax.scatter(centroid[0], centroid[1], label = label, alpha = 1.0, s=1000, facecolor=color, marker="^",
                edgecolor='black', linewidth=1)

plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
save_dir = 'paper_plots'
ax.set_title("LPIPS (AlexNet) Features", fontsize=40, fontweight="bold")
fig.tight_layout()

ax.set_title("LPIPS (AlexNet) Features", fontsize=40, fontweight="bold")
fig.savefig("{}/{}.pdf".format(save_dir, 'lpips_fts_paper'), format="pdf", dpi=200, bbox_inches='tight')

# Plot legend seperately
figsize = (50, 0.1)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)

# add the legend from the previous axes
ax_leg.legend(*ax.get_legend_handles_labels(), loc="upper center", mode = "expand", 
                ncol = 9, frameon=False, fontsize=50, handletextpad=0.01)

# hide the axes frame and the x/y labels
ax_leg.axis('off')
fig_leg.savefig('{}/legend.pdf'.format(save_dir), format='pdf', dpi=1200, bbox_inches='tight')
plt.close()
