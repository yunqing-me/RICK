import argparse
from cgi import test
import math
import random
import os
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
import pickle

from gan_training import utils
from gan_training.eval import Evaluator
from gan_training.utils_model_load import *


# record and visualize the statistics
try:
    import wandb

except ImportError:
    wandb = None


# the same as low-rank probing
from gan_training.models.model_probe_tune import Generator
from gan_training.models.model_probe_tune import Discriminator

from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True,
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def zero_idx_merge(old_zero_dict, new_zero_dict):
    # for each key-value pair in dict, merge idx that zero-mask the value/grad
    merged_dict = dict()
    for key in old_zero_dict:
        merged_dict[key] = np.unique(np.concatenate((old_zero_dict[key], new_zero_dict[key])))
    
    return merged_dict
            


def get_subspace(args, init_z, vis_flag=False):
    std = args.subspace_std
    bs = args.batch if not vis_flag else args.n_sample_store
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z


def train(args, train_loader, generator, discriminator, g_optim, d_optim, g_ema, d_ema, evaluator, device):
    # init the progress bar for visualizing the training process
    pbar = range(args.iter+10)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = generator
    d_module = discriminator
    g_ema_module = g_ema.module
    d_ema_module = d_ema.module

    accum = 0.5 ** (32 / (10 * 1000))  ## 
    ada_augment = torch.tensor([0.0, 0.0], device=device)  ## non-leaking augmentation
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0
    best_fid = 1000 # init with a high number

    # the following defines the constant noise used for generating images at different stages of training
    # sample_z = torch.randn(args.n_sample_store, args.latent, device=device)
    sample_z = torch.load('./noise.pt').cuda()

    # start training
    torch.cuda.empty_cache()
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter+10:
            print("Finishing the training...")
            os.remove(os.path.join(args.output_path, "real_imgs.npy"))
            break
        
        # ----------------- warm-up stage grad ----------------- #
        if i < args.warmup_iter:
            # D
            for name, param in discriminator.named_parameters():
                if name.find('final') >= 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # D
            requires_grad(discriminator, True) 
                
        # regularly estimate fisher, select high FIM Filters
        if ((i-args.warmup_iter) % (args.fisher_freq)) == 0 and (i-args.warmup_iter >= 0) :      
            requires_grad(g_ema, True)
            requires_grad(d_ema, True)
            g_ema.eval()
            d_ema.eval()

            # init fisher dict
            filter_fisher_g = dict()
            filter_fisher_d = dict()

            print("entering evaluation of fisher information...")
            for j in tqdm(range(args.num_fisher_img)): 
                # 0) load a batch of noise and real image, and compute FIM image by image
                noise_fisher    = torch.load(f'../../_noise/{str(j).zfill(4)}.pt').cuda()
                real_img_fisher = next(train_loader).to(device)

                for fisher_idx in range((noise_fisher.size()[0])):
                    g_ema.zero_grad()
                    d_ema.zero_grad()

                    # 1) 
                    # Obtain predicted results
                    fake_img_fisher,  _ = g_ema([(noise_fisher.data)[fisher_idx].view(1,-1)])
                    batch_1_real_img    = (real_img_fisher.data)[fisher_idx].view(1,3,256,256)

                    fake_pred_fisher, _ = d_ema(fake_img_fisher)
                    real_pred_fisher, _ = d_ema(batch_1_real_img)
                
                    # Obtain generator loss/discriminator loss of a single fake/real image
                    g_loss_fisher       = g_nonsaturating_loss(fake_pred_fisher)
                    d_loss_fisher       = d_logistic_loss(real_pred_fisher, fake_pred_fisher)  

                    # 2) Estimate the fisher information and grad of each parameter
                    _, est_fisher_info_g   = g_ema_module.estimate_fisher(loglikelihood=g_loss_fisher)
                    _, est_fisher_info_d   = d_ema_module.estimate_fisher(loglikelihood=d_loss_fisher)

                    # FIM
                    # Record FIM in G
                    for key in est_fisher_info_g:
                        if j == 0 and fisher_idx == 0:
                            filter_fisher_g[key]  = est_fisher_info_g[key].detach().cpu().numpy()
                        else:
                            filter_fisher_g[key] += est_fisher_info_g[key].detach().cpu().numpy()
                    
                    # Record FIM in D
                    for key in est_fisher_info_d:
                        if j == 0 and fisher_idx == 0:
                            filter_fisher_d[key]  = est_fisher_info_d[key].detach().cpu().numpy()
                        else:
                            filter_fisher_d[key] += est_fisher_info_d[key].detach().cpu().numpy()
            
            # avg
            for key in filter_fisher_g:
                filter_fisher_g[key]  /= (args.num_fisher_img * args.batch) 
            for key in filter_fisher_d:
                filter_fisher_d[key]  /= (args.num_fisher_img * args.batch) 

            # ---------------------------- estimate fisher --------------------------- #
            # ------------------------------------------------------------------------ #

            # ------------------------ Screen high FIM Filters ----------------------- #
            # ------------------------------------------------------------------------ #

            # Obtain the quantile values for FC and Conv Layers
            # G: Conv
            grouped_fim_conv_g = []
            filter_fisher_g_interst = dict()
            for block_idx in range(12):
                conv_weight_fim    = filter_fisher_g[f'convs.{block_idx}.conv.weight'].mean(axis=(0,2,3,4))
                grouped_fim_conv_g = np.concatenate((grouped_fim_conv_g, conv_weight_fim), axis=None)
                filter_fisher_g_interst[f'convs.{block_idx}.conv.weight'] = filter_fisher_g[f'convs.{block_idx}.conv.weight']
            cutline_g_conv   = np.percentile(grouped_fim_conv_g, q=args.fisher_quantile)
            pruneline_g_conv = np.percentile(grouped_fim_conv_g, q=args.prune_quantile)
            
            # G: FC
            grouped_fim_fc_g = []
            for block_idx in range(12):
                fc_weight_fim = filter_fisher_g[f'convs.{block_idx}.conv.modulation.weight'].mean(axis=1)
                fc_bias_fim   = filter_fisher_g[f'convs.{block_idx}.conv.modulation.bias']
                fc_fim        = (fc_weight_fim + fc_bias_fim) / 2
                grouped_fim_fc_g = np.concatenate((grouped_fim_fc_g, fc_fim), axis=None)
                
                filter_fisher_g_interst[f'convs.{block_idx}.conv.modulation.weight'] = filter_fisher_g[f'convs.{block_idx}.conv.modulation.weight']
                filter_fisher_g_interst[f'convs.{block_idx}.conv.modulation.bias']   = filter_fisher_g[f'convs.{block_idx}.conv.modulation.bias']
            cutline_g_fc   = np.percentile(grouped_fim_fc_g, q=args.fisher_quantile)
            pruneline_g_fc = np.percentile(grouped_fim_fc_g, q=args.prune_quantile)

            # Decisions
            idx_freeze_g = dict()
            idx_ft_g     = dict()
            idx_prune_g  = dict()
            
            for key in filter_fisher_g_interst:
                if 'modulation' not in key and 'weight' in key: 
                    # for Conv layer
                    conv_weight_fim    = filter_fisher_g_interst[key].mean(axis=(0,2,3,4))
                    
                    # apply heuristics
                    idx_freeze_g[key] = np.where(conv_weight_fim  >  cutline_g_conv)[0]
                    idx_ft_g[key]     = np.where((conv_weight_fim >  pruneline_g_conv) & (conv_weight_fim <= cutline_g_conv))[0]
                    idx_prune_g[key]  = np.where(conv_weight_fim  <= pruneline_g_conv)[0]

                elif 'modulation' in key and 'weight' in key:  
                    # for FC layer
                    fc_weight_fim = filter_fisher_g_interst[key].mean(axis=1)
                    fc_bias_fim   = filter_fisher_g_interst[key.replace('weight', 'bias')]
                    fc_fim        = (fc_weight_fim + fc_bias_fim) / 2
                    
                    # apply heuristics
                    # FC weights
                    idx_freeze_g[key] = np.where(fc_fim  >  cutline_g_fc)[0]
                    idx_ft_g[key]     = np.where((fc_fim >  pruneline_g_fc) & (fc_fim <= cutline_g_fc))[0]
                    idx_prune_g[key]  = np.where(fc_fim  <= pruneline_g_fc)[0]
                    # FC bias
                    idx_freeze_g[key.replace('weight', 'bias')] = np.where(fc_fim  >  cutline_g_fc)[0]
                    idx_ft_g[key.replace('weight', 'bias')]     = np.where((fc_fim >  pruneline_g_fc) & (fc_fim <= cutline_g_fc))[0]
                    idx_prune_g[key.replace('weight', 'bias')]  = np.where(fc_fim  <= pruneline_g_fc)[0]
                
            # Obtain the quantile values for Conv Layers
            # D: Conv
            grouped_fim_conv_d = []
            filter_fisher_d_interst = dict()
            for block_idx in range(1,7):
                # for normal layers
                for layer_idx in range(2):
                    conv_weight_fim = filter_fisher_d[f'convs.{block_idx}.conv{layer_idx+1}.{layer_idx}.weight'].mean(axis=(1,2,3))       
                    conv_bias_fim   = filter_fisher_d[f'convs.{block_idx}.conv{layer_idx+1}.{layer_idx+1}.bias']      
                    conv_fim      = (conv_weight_fim + conv_bias_fim) / 2
                    grouped_fim_conv_d = np.concatenate((grouped_fim_conv_d, conv_fim), axis=None)
                    
                    filter_fisher_d_interst[f'convs.{block_idx}.conv{layer_idx+1}.{layer_idx}.weight'] = filter_fisher_d[f'convs.{block_idx}.conv{layer_idx+1}.{layer_idx}.weight']
                    filter_fisher_d_interst[f'convs.{block_idx}.conv{layer_idx+1}.{layer_idx+1}.bias'] = filter_fisher_d[f'convs.{block_idx}.conv{layer_idx+1}.{layer_idx+1}.bias']
                    
                    # for skip layers
                    if layer_idx == 1:
                        conv_skip_fim    = filter_fisher_d[f'convs.{block_idx}.skip.{layer_idx}.weight'].mean(axis=(1,2,3))
                        grouped_fim_conv_d = np.concatenate((grouped_fim_conv_d, conv_skip_fim), axis=None)
                        filter_fisher_d_interst[f'convs.{block_idx}.skip.{layer_idx}.weight'] = filter_fisher_d[f'convs.{block_idx}.skip.{layer_idx}.weight']
            cutline_d_conv   = np.percentile(grouped_fim_conv_d, q=args.fisher_quantile)
            pruneline_d_conv = np.percentile(grouped_fim_conv_d, q=args.prune_quantile)

            # Obtain decisions for D
            idx_freeze_d = dict()
            idx_ft_d     = dict()
            idx_prune_d  = dict()
            for key in filter_fisher_d_interst:
                if 'skip' not in key and 'weight' in key:
                    # resemble FIM
                    conv_weight_fim = filter_fisher_d_interst[key].mean(axis=(1,2,3))
                    conv_bias_fim   = filter_fisher_d_interst[key.replace(f'{key[-8]}.weight', f'{str(int(key[-8])+1)}.bias')]
                    conv_fim        = (conv_weight_fim + conv_bias_fim) / 2

                    # apply heuristics
                    # for normal layer-weights
                    idx_freeze_d[key] = np.where(conv_fim  >  cutline_d_conv)[0]
                    idx_ft_d[key]     = np.where((conv_fim >  pruneline_d_conv) & (conv_fim <=  cutline_d_conv))[0]
                    idx_prune_d[key]  = np.where(conv_fim  <= pruneline_d_conv)[0]
                    
                    # for normal layer-bias
                    idx_freeze_d[key.replace(f'{key[-8]}.weight', f'{str(int(key[-8])+1)}.bias')] = np.where(conv_fim  >  cutline_d_conv)[0]
                    idx_ft_d[key.replace(f'{key[-8]}.weight', f'{str(int(key[-8])+1)}.bias')]     = np.where((conv_fim >  pruneline_d_conv) & (conv_fim <=  cutline_d_conv))[0]
                    idx_prune_d[key.replace(f'{key[-8]}.weight', f'{str(int(key[-8])+1)}.bias')]  = np.where(conv_fim  <= pruneline_d_conv)[0]

                elif 'skip' in key and 'weight' in key:
                    # resemble FIM
                    conv_skip_fim    = filter_fisher_d_interst[key].mean(axis=(1,2,3))

                    # apply heuristics
                    idx_freeze_d[key] = np.where(conv_skip_fim  >  cutline_d_conv)[0]
                    idx_ft_d[key]     = np.where((conv_skip_fim >= pruneline_d_conv) & (conv_skip_fim <= cutline_d_conv))[0]
                    idx_prune_d[key]  = np.where(conv_skip_fim  <  pruneline_d_conv)[0]
        
            if i == args.warmup_iter:
                # init zero dicts
                zero_filter_idx_g = idx_prune_g
                zero_filter_idx_d = idx_prune_d
            else:
                # update zero dicts
                zero_filter_idx_g = zero_idx_merge(zero_filter_idx_g, idx_prune_g)
                zero_filter_idx_d = zero_idx_merge(zero_filter_idx_d, idx_prune_d)
            
        # ---------------------------- Dynamic training -------------------------- #
        real_img = next(train_loader)
        real_img = real_img.to(device)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(noise)

        if args.augment:
            real_img, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img)
        real_pred, _ = discriminator(
            real_img)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        # only update D
        discriminator.zero_grad()
        d_loss.backward()

        if i < args.warmup_iter:
            # warmup stage adaptation
            d_optim.step()
            torch.cuda.empty_cache()
        else:
            # ----------------------------------- dynamic filter-adaptation ------------------------------- #
            # D: zero-out grad for filters with high FIM
            for name, param in discriminator.module.named_parameters():
                if name in idx_freeze_d.keys():
                    param.grad[idx_freeze_d[name]] = 0
                with torch.no_grad():
                    if name in zero_filter_idx_d.keys():
                        param[zero_filter_idx_d[name]] = 0
                        param.grad[zero_filter_idx_d[name]] = 0
            # -----------------------------------------------------------------------------------------------
            d_optim.step()
            torch.cuda.empty_cache()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        # using r1_loss to regularize the D, for every 16 iterations
        d_regularize = i % args.d_reg_every == 0 

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img)
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            # warmup stage adaptation
            if i < args.warmup_iter:
                d_optim.step()
                torch.cuda.empty_cache()
            else:
                # ----------------------------------- dynamic filter-adaptation ------------------------------- #
                # D: zero-out grad for filters with high FIM
                for name, param in discriminator.module.named_parameters():
                    if name in idx_freeze_d.keys():
                        param.grad[idx_freeze_d[name]] = 0
                    with torch.no_grad():
                        if name in zero_filter_idx_d.keys():
                            param[zero_filter_idx_d[name]] = 0
                            param.grad[zero_filter_idx_d[name]] = 0
                # -----------------------------------------------------------------------------------------------
                d_optim.step()      
                torch.cuda.empty_cache()  

        loss_dict["r1"] = r1_loss

        # adversarial training G, no update D
        # requires_grad(generator, True)
        # requires_grad(discriminator, False)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        g_loss = g_loss

        loss_dict["g"] = g_loss

        # only update G
        generator.zero_grad()

        if i < args.warmup_iter:
            pass
        else:
            g_loss.backward()
            # -------------------------------------------------------------
            # G: zero-out grad for high FIM filters
            for name, param in generator.named_parameters():
                if name in idx_freeze_g.keys():
                    if param.ndim != 5:
                        param.grad[idx_freeze_g[name]] = 0
                    else:
                        param.grad[:, idx_freeze_g[name], :, :, :] = 0
                with torch.no_grad():
                    if name in zero_filter_idx_g.keys():
                        if param.ndim != 5:
                            param[zero_filter_idx_g[name]] = 0
                            param.grad[zero_filter_idx_g[name]] = 0
                        else:
                            param[:, zero_filter_idx_g[name], :, :, :] = 0
                            param.grad[:, zero_filter_idx_g[name], :, :, :] = 0
            # -------------------------------------------------------------
            g_optim.step()
            torch.cuda.empty_cache()

        # to save up space
        # del rel_loss, g_loss, d_loss, fake_img, fake_pred, real_img, real_pred, anchor_feat, compare_feat, dist_source, dist_target, feat_source, feat_target
        del g_loss, d_loss, fake_img, fake_pred, real_img, real_pred

        g_regularize = i % args.g_reg_every == 0
        if g_regularize and i >= args.warmup_iter:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            if i < args.warmup_iter:
                pass
            else:
                weighted_path_loss.backward()
                # ----------------------------------------------------------------
                # G: zero-out grad for high FIM filters
                for name, param in generator.named_parameters():
                    if name in idx_freeze_g.keys():
                        if param.ndim != 5:
                            param.grad[idx_freeze_g[name]] = 0
                        else:
                            param.grad[:, idx_freeze_g[name], :, :, :] = 0
                    with torch.no_grad():
                        if name in zero_filter_idx_g.keys():
                            if param.ndim != 5:
                                param[zero_filter_idx_g[name]] = 0
                                param.grad[zero_filter_idx_g[name]] = 0
                            else:
                                param[:, zero_filter_idx_g[name], :, :, :] = 0
                                param.grad[:, zero_filter_idx_g[name], :, :, :] = 0
                # -----------------------------------------------------------------
                g_optim.step()
                torch.cuda.empty_cache()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()

        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if not (i % args.eval_in_training_freq == 0):
                if wandb and args.wandb:
                    wandb.log(
                        {
                            "Generator": g_loss_val,
                            "Discriminator": d_loss_val,
                            "Augment": ada_aug_p,
                            "Rt": r_t_stat,
                            "R1": r1_val,
                            "Path Length Regularization": path_loss_val,
                            "Mean Path Length": mean_path_length,
                            "Real Score": real_score_val,
                            "Fake Score": fake_score_val,
                            "Path Length": path_length_val,
                        }
                    )

            # 1) generate intermidiate images
            if i % args.samples_freq == 0:
                if args.store_samples:
                    with torch.set_grad_enabled(False):
                        g_ema.eval()
                        sample, _ = g_ema([sample_z.data])
                        utils.save_images(
                            sample,
                            f"%s/{str(i).zfill(6)}.png" % (args.sample_dir),
                            nrow=int(args.n_sample_store ** 0.5)
                        )
                        del sample

            #  2) save intermediate checkpoints 
            if (i % args.checkpoints_freq == 0) and (i > 0):
                if args.store_checkpoints:
                    torch.save(
                        {
                            "g_ema": g_ema.state_dict(),
                            # uncomment the following lines only if you wish to resume training after saving. 
                            # Otherwise, saving just the generator is sufficient for evaluations

                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                        },
                        f"%s/{str(i).zfill(6)}.pt" % (args.checkpoint_dir),
                    )
                else:
                    pass
            
            # 3) evaluation in training
            if args.eval_in_training:
                if (i % args.eval_in_training_freq == 0):
                    with torch.no_grad():
                        # eval metrics
                        score = evaluator.compute_inception_score(kid=False, pr=False)
                        torch.cuda.empty_cache()
                        if score['fid'] < best_fid:
                            best_fid = score['fid']
                            torch.save(
                                {
                                    "g_ema": g_ema.state_dict(),
                                    # uncomment the following lines only if you wish to resume training after saving. 
                                    # Otherwise, saving just the generator is sufficient for evaluations

                                    "g": g_module.state_dict(),
                                    "d": d_module.state_dict(),
                                    "g_optim": g_optim.state_dict(),
                                    "d_optim": d_optim.state_dict(),
                                },
                                os.path.join(args.checkpoint_dir, "best.pt"),
                            )
                            np.savetxt(os.path.join(args.checkpoint_dir, 'best_fid.txt'), score['fid'].reshape(1, -1))
                        torch.cuda.empty_cache()
                    if wandb and args.wandb:
                        wandb.log(
                            {  
                                "FID"    : score['fid'],
                                "Generator": g_loss_val,
                                "Discriminator": d_loss_val,
                            }
                        )

            # 5) update ema generator 
            accumulate(g_ema_module, g_module, accum)
            accumulate(d_ema_module, d_module.module, accum)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='tmp')
    parser.add_argument("--data_path", type=str, default='babies')
    parser.add_argument("--iter", type=int, default=31)
    parser.add_argument("--highp", type=int, default=1)
    parser.add_argument("--subspace_freq", type=int, default=4)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--feat_const_batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256, help="size of the img, must be square")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_res", type=int, default=128)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--subspace_std", type=float, default=0.05)
    parser.add_argument("--ckpt_source", type=str, default="style_gan_source_ffhq.pt", help="pretrained model")
    parser.add_argument("--source_key", type=str, default='ffhq')
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", dest='augment', action='store_true')
    parser.add_argument("--no-augment", dest='augment', action='store_false')
    parser.add_argument("--augment_p", type=float, default=0.0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--n_sample_train", type=int, default=10)
    parser.add_argument("--n_sample_store", type=int, default=25, help="# of generated images using intermediate models")
    parser.add_argument("--n_sample_test", type=int, default=25)
    
    parser.add_argument("--store_checkpoints", action="store_true")
    parser.add_argument("--store_samples", action="store_true")
    parser.add_argument("--eval_in_training", action="store_true")

    parser.add_argument("--num_fisher_img", type=int, default=5)
    parser.add_argument("--fisher_freq", type=int, default=2)
    parser.add_argument("--fisher_coef", type=float, default=1.0)
    parser.add_argument("--fisher_quantile", type=float, default=75)
    parser.add_argument("--prune_quantile", type=float, default=0.1) # cutline for zero-out filters
    parser.add_argument("--warmup_iter", type=int, default=10)  # warm up for D final layers

    parser.add_argument("--checkpoints_freq", type=int, default=500)
    parser.add_argument("--samples_freq", type=int, default=500)
    parser.add_argument("--eval_in_training_freq", type=int, default=500)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='debug')
    parser.add_argument("--wandb_run_name", type=str, default='debug')

    parser.add_argument("--method", type=str, default='dynamic_1')

    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    # Step 1. Pre-experiment setups
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if get_rank() == 0:
        print("Basic setups:", '\n', args)

    # reset directory
    args.output_path    = os.path.join('../../_output_style_gan/', args.exp)
    args.sample_dir     = os.path.join('../../_output_style_gan/', args.exp, 'samples')
    args.checkpoint_dir = os.path.join('../../_output_style_gan/', args.exp, 'checkpoints')

    # Create missing directories
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir, exist_ok=True)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    # pre-process the dataset (data is resized and binarized into mdb file)
    transform_train = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    # define datasets and loaders
    data_path_train = os.path.join('../../_processed_train', args.data_path) # only for 10-shot
    data_path_test   = os.path.join('../../_processed_test', args.data_path)

    if args.n_sample_train == 10:
        train_dataset   = MultiResolutionDataset(data_path_train, transform_train, args.size)
    else:
        train_dataset   = MultiResolutionDataset(data_path_test, transform_train, args.size)
        # few_shot_idx    = np.random.randint(0, train_dataset.length, size=args.n_sample_train)
        few_shot_idx    = np.random.choice(train_dataset.length, size=args.n_sample_train, replace=False)
        np.savetxt(f"./{args.output_path}/{args.n_sample_train}-shot-index.txt", few_shot_idx)
        train_dataset   = data.Subset(train_dataset, indices=few_shot_idx)
        print(f"Few-shot transfer with {few_shot_idx.size}-shot images")
    train_loader   = data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=False),
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = sample_data(train_loader)

    test_dataset   = MultiResolutionDataset(data_path_test, transform_test, args.size)
    test_loader    = data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        sampler=data_sampler(test_dataset, shuffle=False, distributed=False),
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    test_loader = sample_data(test_loader)

    # save the args
    argsDict = args.__dict__
    with open(os.path.join(args.output_path, 'args.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    
    # save the training script
    import shutil
    my_file = './train_dynamic_update_prune.py'
    to_file = os.path.join(args.output_path, "./train_script.py")
    shutil.copy(str(my_file), str(to_file))

    # Load pretrained GAN models
    if get_rank() == 0 and wandb and args.wandb:
        run = wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)

    # initialize the models using styleGAN2
    generator     = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)  
    g_ema         = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    d_ema         = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    g_dict = generator.state_dict()
    d_dict = discriminator.state_dict()
    if args.ckpt_source is not None:
        ckpt_source_path = os.path.join("../../_pretrained/", args.ckpt_source)
        print("load model:", args.ckpt_source)
        assert args.source_key in args.ckpt_source
        ckpt_source = torch.load(ckpt_source_path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt_source["g"], strict=False)
        g_ema.load_state_dict(ckpt_source["g_ema"], strict=False)
        discriminator.load_state_dict(ckpt_source["d"], strict=False)
        d_ema.load_state_dict(ckpt_source["d"], strict=False)
    
    # ---------------------------------------------------------------------- #

    ## final generated results
    g_ema.eval()
    d_ema.eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    ## --------------- Specify trainable parameters ---------------- ##
    ## ------------------------------------------------------------- ##

    #                             original version
    #
    # g_optim = optim.Adam(
    #     generator.parameters(),
    #     lr=args.lr * g_reg_ratio,
    #     betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    # )
    # d_optim = optim.Adam(
    #     discriminator.parameters(),
    #     lr=args.lr * d_reg_ratio,
    #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    # )

    #                               new version for G
    # 
    g_probe_params = []
    for name, param in generator.named_parameters():
        if 'convs' in name: 
            g_probe_params.append(param)

    g_optim = optim.Adam(
        g_probe_params,
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    #                               new version for D
    #
    d_probe_params = []
    for name, param in discriminator.named_parameters():
        if 'convs' in name and 'convs.0' not in name:  
            d_probe_params.append(param)
        elif 'final' in name:
            d_probe_params.append(param)
    d_optim = optim.Adam(
        d_probe_params,
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    ## --------------- Specify trainable parameters ---------------- ##
    ## ------------------------------------------------------------- ##
    # print the number of trainable parameters
    # get_parameter_number(generator, name=f'Generator-filter-level-init')
    # get_parameter_number(discriminator, name=f'Discriminator-filter-level-init')
    
    # ---------------------------------------------------------------------- #
    
    geneator = nn.parallel.DataParallel(generator)
    g_ema    = nn.parallel.DataParallel(g_ema)
    discriminator = nn.parallel.DataParallel(discriminator)
    d_ema         = nn.parallel.DataParallel(d_ema)

    # define the evaluator
    if get_rank() == 0:
        if args.exp == 'tmp' and not args.eval_in_training:
            x_real_test = None
        else:
            x_real_test = utils.get_nsamples_lmdb(test_loader, args.n_sample_test, set_len=test_dataset.length)
        # to compute IS and FID
        evaluator = Evaluator(args, g_ema,
                            batch_size=args.batch, 
                            device=device,
                            fid_real_samples=x_real_test, 
                            inception_nsamples=args.n_sample_test,
                            fid_sample_size=args.n_sample_test)
        x_real  = utils.get_nsamples_lmdb(train_loader, 10)
        utils.save_images(x_real, os.path.join(args.output_path, 'real.png'), nrow=5)
        del x_real_test

    # 4. training.
    train(args, train_loader, generator, discriminator, g_optim, d_optim, g_ema, d_ema, evaluator, device)