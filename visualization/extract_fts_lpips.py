import math
import random
import functools
import operator
from turtle import pen

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import torchvision

from model import Discriminator
from torchvision import transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader
import os, PIL
from fid import calc_fid
import numpy as np
import pickle
from scipy import linalg
from calc_inception import load_patched_inception_v3
from tqdm import tqdm

import lpips

# Keep penultimate features as global varialble such that hook modifies these features
penultimate_fts = None


def get_penultimate_fts(self, input, output):
    global penultimate_fts
    #print(input)
    penultimate_fts = output
    return None


def get_lpips_net():
    device = "cuda:0"

    # LPIPS model
    percept = lpips.PerceptualLoss(
        model="net-lin", net="alex", use_gpu=device.startswith("cuda")
    )

    print(percept)
    return percept.model.net


class CustomDataset_LPIPS(Dataset):
    def __init__(self, dir_path, transform=None, num_samples=None, repeat=False):
        super(CustomDataset_LPIPS, self).__init__()
        self.dir_path = dir_path
        self.all_image_paths = [os.path.join(self.dir_path, i) for i in os.listdir(self.dir_path)][:num_samples]

        if repeat:
            self.all_image_paths = self.all_image_paths*2

        self.transform = transform


    def __getitem__(self, idx):
        img_path = self.all_image_paths[idx]
        #img = PIL.Image.open(img_path).convert('RGB')

        img = lpips.load_image(img_path)
        img = lpips.im2tensor(img)
        #print(img.size())

        if self.transform is not None:
            return self.transform(img)

        return img


    def __len__(self):
        return len(self.all_image_paths)


def create_image_dataloader(path, num_samples):

    # LPIPS trained with 64 x 64 patches, therefore resize appropriately
    # If even repeated with different sizes, the results have similar trends.
    transform = transforms.Compose([
            transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(64), # Use CenterCrop / Similar trend obtained even if resized + crop
            #transforms.ToTensor(),
            #normalize,
        ])

    ds = CustomDataset_LPIPS(path, transform, num_samples)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    return loader



def extract_embeddings(lpips_model, dl):
    global penultimate_fts
    lpips_model.eval()
    lpips_model.features[12].register_forward_hook(get_penultimate_fts)

    all_embeddings = None

    with tqdm(len(dl)) as pbar:
        for batch_idx, x in enumerate(dl):
            x = x.cuda()
            penultimate_fts = None

            # =================== extract penultimate layer features =======================
            # Register hook to avg pool layer
            with torch.no_grad():
                #print(x.size())
                output = lpips_model(x)
                penultimate_fts = penultimate_fts.view(penultimate_fts.size(0), -1)
                #print(penultimate_fts.size())
                assert torch.is_tensor(penultimate_fts)

                if all_embeddings is None:
                    all_embeddings = penultimate_fts.detach().cpu()
                else:
                    all_embeddings = torch.cat( (all_embeddings, penultimate_fts.detach().cpu() ), 0)
                
            pbar.set_description("current extract: ({}, {}) | total extract: ({}, {})".format( penultimate_fts.size(0), penultimate_fts.size(1),
                             all_embeddings.size(0), all_embeddings.size(1) ) )
            pbar.update(1)

    return all_embeddings


def save_embeddings_as_pt(features, name):
    dir_name = 'embeddings_lpips'
    os.makedirs(dir_name, exist_ok=True)
    torch.save(features, os.path.join(dir_name, '{}.pt'.format(name)) ) 


def load_embeddings(name):
    embeddings = torch.load('embeddings_lpips/{}.pt'.format(name))
    print("Loaded {} embeddings with size = {} ".format(name, embeddings.size()))
    return embeddings



def main(dataset_name, num_samples=None):
    dataset_paths = {
        'ffhq' : '/mnt/data/few-shot-modulation/source/FFHQ_256/',

        # These datasets were used in FID Analysis
        'lsun_bedroom' : '/mnt/data/few-shot-modulation/benchmarks/lsun_bedroom/',
        'lsun_cat' : '/mnt/data/few-shot-modulation/benchmarks/lsun_cat/',
        'celeba-hq': '/mnt/data/few-shot-modulation/benchmarks/celeba-hq-imgs/',

        # Close Proximity Domains for FFHQ
        'babies' : '/mnt/data/few-shot-modulation/cr/babies/images/',
        'sunglasses' : '/mnt/data/few-shot-modulation/cr/sunglasses/images/',
        'metfaces' : '/mnt/data/few-shot-modulation/cr/metfaces/images/',
        'sketches' : '/mnt/data/few-shot-modulation/cr/sketches/images/',

        # Distant Target Domains for FFHQ
        'afhq_cat' : '/mnt/data/few-shot-modulation/lr/afhq_cat/images/',
        'afhq_dog' : '/mnt/data/few-shot-modulation/lr/afhq_dog/images/',
        'afhq_wild' : '/mnt/data/few-shot-modulation/lr/afhq_wild/images/',

    }


    device = "cuda:0"
    #lpips_model = get_lpips_net()
    lpips_model = torchvision.models.alexnet(pretrained=True).to(device)
    lpips_model.eval()

    # Extract ffhq embeddings
    dl = create_image_dataloader(dataset_paths[dataset_name], num_samples)
    ffhq_embeddings = extract_embeddings(lpips_model, dl)
    save_embeddings_as_pt(ffhq_embeddings, dataset_name)
    ffhq_embeddings = load_embeddings(dataset_name)



if __name__ == '__main__':
    # Extract inception embeddings for all datasets used in Main Paper
    dataset_names = [
        'ffhq',
        'sunglasses',
        'babies',
        'metfaces',

        'afhq_cat',
        'afhq_dog',
        'afhq_wild',

    ]

    for dataset_name in dataset_names:
        main(dataset_name)