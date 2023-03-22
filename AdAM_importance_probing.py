import argparse
from cgi import test
import math
import random
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
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



from gan_training.models.model_adam import Generator as Generator
from gan_training.models.model_adam  import Discriminator as Discriminator

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


def get_subspace(args, init_z, vis_flag=False):
    std = args.subspace_std
    bs = args.batch if not vis_flag else args.n_sample_store
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z


def calculate_fisher(args, train_loader, generator, discriminator, g_optim, d_optim, g_ema, d_ema, device):
    pbar = range(args.fisher_iter+5)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=0,
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
    for idx in pbar:
        i = idx
        # --------------- --------------- ----------------- #
        # --------------- --------------- ----------------- #
        # --------------- estimate fisher ----------------- #

        if (i % args.fisher_freq == 0):      
            requires_grad(g_ema, True)
            requires_grad(d_ema, True)
            g_ema.eval()
            d_ema.eval()
            # init fisher dict
            filter_grad_g = dict()
            filter_fisher_g = dict()

            filter_grad_d = dict()          
            filter_fisher_d = dict()

            print("entering evaluation of fisher information...")
            for j in tqdm(range(args.num_batch_fisher)):  # for each iteration, we read 4 noise input but calculate FIM one-by-one
                # 0) load a batch of noise and real image, and compute FIM image by image
                noise_fisher    = torch.load(f'./_noise/{str(j).zfill(4)}.pt').cuda()
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
                    g_grads, est_fisher_info_g   = g_ema_module.estimate_fisher(loglikelihood=g_loss_fisher)
                    d_grads, est_fisher_info_d   = d_ema_module.estimate_fisher(loglikelihood=d_loss_fisher)

                    # Grad
                    # store grad for G
                    for k, (n, p) in enumerate(g_ema_module.named_parameters()):
                        if p.requires_grad:
                            if g_grads[k] is not None:
                                if j == 0 and fisher_idx == 0:
                                    filter_grad_g[n] =  g_grads[k].detach()
                                else:
                                    filter_grad_g[n] += g_grads[k].detach()

                    # store grad for D
                    for k, (n, p) in enumerate(d_ema_module.named_parameters()):
                        if p.requires_grad:
                            if d_grads[k] is not None:
                                if j == 0 and fisher_idx == 0:
                                    filter_grad_d[n] =  d_grads[k].detach()
                                else:
                                    filter_grad_d[n] += d_grads[k].detach()

                    # FIM
                    # Record filter-level FIM in G
                    for key in est_fisher_info_g:
                        if j == 0 and fisher_idx == 0:
                            filter_fisher_g[key]  = est_fisher_info_g[key].detach().cpu().numpy()
                        else:
                            filter_fisher_g[key] += est_fisher_info_g[key].detach().cpu().numpy()
                    
                    # Record filter-level FIM in D
                    for key in est_fisher_info_d:
                        if j == 0 and fisher_idx == 0:
                            filter_fisher_d[key]  = est_fisher_info_d[key].detach().cpu().numpy()
                        else:
                            filter_fisher_d[key] += est_fisher_info_d[key].detach().cpu().numpy()
            
            # avg
            for key in filter_grad_g:
                filter_grad_g[key]    /= (args.num_batch_fisher * args.batch) 
            for key in filter_grad_d:
                filter_grad_d[key]    /= (args.num_batch_fisher * args.batch) 

            for key in filter_fisher_g:
                filter_fisher_g[key]  /= (args.num_batch_fisher * args.batch) 
            for key in filter_fisher_d:
                filter_fisher_d[key]  /= (args.num_batch_fisher * args.batch) 

            torch.save(filter_grad_g, os.path.join(args.checkpoint_dir, "filter_grad_g.pt"))        
            torch.save(filter_fisher_g, os.path.join(args.checkpoint_dir, "filter_fisher_g.pt"))

            torch.save(filter_grad_d, os.path.join(args.checkpoint_dir, "filter_grad_d.pt"))
            torch.save(filter_fisher_d, os.path.join(args.checkpoint_dir, "filter_fisher_d.pt"))

        # --------------- estimate fisher ----------------- #
        # --------------- --------------- ----------------- #
        # --------------- --------------- ----------------- #

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
        d_optim.step()

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

            d_optim.step()
        loss_dict["r1"] = r1_loss

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
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        # to save up space
        del g_loss, d_loss, fake_img, fake_pred, real_img, real_pred

        if g_regularize:
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

            weighted_path_loss.backward()

            g_optim.step()

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

        # 4) update ema GAN
        accumulate(g_ema_module, g_module, accum)  # store the moving average parameters in g_ema
        accumulate(d_ema_module, d_module.module, accum)  # store the moving average parameters in d_ema


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='tmp')
    parser.add_argument("--data_path", type=str, default='babies')
    parser.add_argument("--iter", type=int, default=1001)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256, help="size of the img, must be square")
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
    parser.add_argument("--n_sample_test", type=int, default=5000)
    
    parser.add_argument("--num_batch_fisher", type=int, default=5)
    parser.add_argument("--fisher_freq", type=int, default=10)
    parser.add_argument("--fisher_iter", type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    # --------------------------------- # 
    # Step 1. Pre-experiment setups
    # --------------------------------- # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if get_rank() == 0:
        print("Basic setups:", '\n', args)

    args.output_path    = os.path.join('./_output_style_gan/', args.exp)
    args.checkpoint_dir = os.path.join('./_output_style_gan/', args.exp, 'checkpoints')

    # Create missing directories
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    ## ------------------------- Modulate all blocks for estimating FIM ---------------------------- ##

    # initialize the models using style_gan2, with KML Module
    generator     = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)  
    g_ema         = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    d_ema         = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_dict = generator.state_dict()
    d_dict = discriminator.state_dict()
    if args.ckpt_source is not None:
        ckpt_source_path = os.path.join("./_pretrained/", args.ckpt_source)
        print("load model:", args.ckpt_source)
        assert args.source_key in args.ckpt_source
        ckpt_source = torch.load(ckpt_source_path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt_source["g"], strict=False)
        g_ema.load_state_dict(ckpt_source["g_ema"], strict=False)
        discriminator.load_state_dict(ckpt_source["d"], strict=False)
        d_ema.load_state_dict(ckpt_source["d"], strict=False)

    # trainable parameters in G
    for name, param in generator.named_parameters():
        if name.find('u_vector') >= 0:
            param.requires_grad = True
        elif name.find('v_vector') >= 0:
            param.requires_grad = True
        elif name.find('b_vector') >= 0:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # key-words of trainable parameters in D
    d_fine_tune = ['final', 'u_vector', 'v_vector', 'b_vector']

    for name, param in discriminator.named_parameters():
        d_flag = 0
        for key in d_fine_tune:
            if key in name:
                param.requires_grad = True
                d_flag += 1
        if d_flag == 0:
            param.requires_grad = False

    # print the number of trainable parameters
    get_parameter_number(generator, name=f'Generator-fisher')
    get_parameter_number(discriminator, name=f'Discriminator-fisher')

    ## final generated results
    g_ema.eval()
    d_ema.eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    geneator      = nn.parallel.DataParallel(generator)
    g_ema         = nn.parallel.DataParallel(g_ema)
    discriminator = nn.parallel.DataParallel(discriminator)
    d_ema         = nn.parallel.DataParallel(d_ema)

    # ----------------------------------------------------------------------- #
    # Step 2. pre-process the dataset (resized and binarized into lmdb file)
    # ----------------------------------------------------------------------- #
    transform = transforms.Compose(
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
    data_path_train = os.path.join('./_processed_train', args.data_path) # only for 10-shot
    data_path_test   = os.path.join('./_processed_test', args.data_path)

    if args.n_sample_train <= 10:
        train_dataset   = MultiResolutionDataset(data_path_train, transform, args.size)
    else:
        train_dataset   = MultiResolutionDataset(data_path_test, transform, args.size)
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

    # save the args
    argsDict = args.__dict__
    with open(os.path.join(args.output_path, 'args.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    # save the training script
    import shutil
    my_file = './AdAM_importance_probing.py'
    to_file = os.path.join(args.output_path, "./train_script.py")
    shutil.copy(str(my_file), str(to_file))

    # ----------------------------------------- #
    # Step 3. AdAM: Importance Probing
    # ----------------------------------------- #
    calculate_fisher(args, train_loader, generator, discriminator, g_optim, d_optim, g_ema, d_ema, device)
