<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Few-shot Image Generation via Adaptation-Aware <br> Kernel Modulation</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://scholar.google.com/citations?user=kQA0x9UAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Yunqing Zhao*</a>&nbsp;/&nbsp;
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan Chandrasegaran*</a>&nbsp;/&nbsp;
    <a href="https://miladabd.github.io/" target="_blank" style="text-decoration: none;">Milad Abdollahzadeh*</a>&nbsp;/&nbsp;
    <a href="https://sites.google.com/site/mancheung0407/" target="_blank" style="text-decoration: none;">Ngai&#8209;Man Cheung</a></br>
Singapore University of Technology and Design (<b>SUTD</b>)<br/>
</p>

<p align='center';>
<b>
<em>The Thirty-Sixth Annual Conference on Neural Information Processing Systems (NeurIPS 2022);</em> <br>
<em>Ernest N. Morial Convention Center, New Orleans, LA, USA.</em>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://yunqing-me.github.io/AdAM//" target="_blank" style="text-decoration: none;">Project Page</a>&nbsp;/&nbsp;
    <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202022/d0ac1ed0c5cb9ecbca3d2496ec1ad984.png" target="_blank" style="text-decoration: none;">Poster</a>&nbsp;/&nbsp;
    <a href="https://drive.google.com/file/d/1hNSIlu0zhjGvqq-gG928jIICCCxuhFHz/view?usp=share_link" target="_blank" style="text-decoration: none;">Slides</a>&nbsp;/&nbsp;
    <a href="https://arxiv.org/abs/2210.16559" target="_blank" style="text-decoration: none;">Paper</a>&nbsp;
    <!-- /&nbsp; -->
    <!-- <a href="https://recorder-v3.slideslive.com/?share=74947&s=c88e53c5-a3c2-46c9-9719-092b74eca0c2" target="_blank" style="text-decoration: none;">Talk</a>&nbsp; -->
</b>
</p>


----------------------------------------------------------------------

#### TL, DR: 
```
In this research, we propose Adaptation-Aware Kernel Modulation (AdAM) for few-shot image generation, that aims to identify kernels in source GAN important for target adaptation. 

The model can perform GAN adaptation using very few samples from target domains with different proximity to the source.
```

## Installation and Environment:

- Platform: Linux
- Tesla V100 GPUs with CuDNN 10.1
- PyTorch 1.7.0
- Python 3.6.9
- lmdb, tqdm

Alternatively, you can install all libiraries through:  `pip install -r requirements.txt`

## Analysis of Source ↦ Target distance

We analyze the Source ↦ Target domain relation in the Sec. 3 (and Supplementary). See below for related steps in this analysis.

Step 1. `git clone https://github.com/rosinality/stylegan2-pytorch.git`

Step 2. Move `./visualization` to `./stylegan2-pytorch`

Step 3. Then, refer to the visualization code in `./visualization`.

## Pre-processing for training

### Step 1. 
Prepare the few-shot training dataset using lmdb format

For example, download the 10-shot target set, `Babies` ([Link](https://drive.google.com/file/d/1P8JMLq2Kk61MbEZDgwytqXxfrhG-NqcR/view?usp=sharing)) and `AFHQ-Cat`([Link](https://drive.google.com/file/d/1zgacEE0jiiDxttbK81fk6miY_4Ithhw-/view?usp=sharing)), and organize your directory as follows:

~~~
10-shot-{babies/afhq_cat}
└── images		
    └── image-1.png
    └── image-2.png
    └── ...
    └── image-10.png
~~~

Then, transform to lmdb format:

`python prepare_data.py --input_path [your_data_path_of_{babies/afhq_cat}] --output_path ./_processed_train/[your_lmdb_data_path_of_{babies/afhq_cat}]`

### Step 2. 
Prepare the entire target dataset for evaluation

For example, download the entire dataset, `Babies`([Link](https://drive.google.com/file/d/1xBpBRmPRoVXsWerv_zx4kQ4nDQUOsqu_/view?usp=share_link)) and `AFHQ-Cat`([Link](https://drive.google.com/file/d/1_-cDkzqz3LlotXSYMBXZLterSQe4fR7S/view?usp=share_link)), and organize your directory as follows:

~~~
entire-{babies/afhq_cat}
└── images		
    └── image-1.png
    └── image-2.png
    └── ...
    └── image-n.png
~~~

Then, transform to lmdb format for evaluation

`python prepare_data.py --input_path [your_data_path_of_entire_{babies/afhq_cat}] --output_path ./_processed_test/[your_lmdb_data_path_of_entire_{babies/afhq_cat}]`

### Step 3. 
Download the GAN model pretrained on FFHQ from [here](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view). Then, save it to `./_pretrained/style_gan_source_ffhq.pt`.

### Step 4.
Randomly generate abundant Gaussian noise input (the same dimension as input to the generator) for Importance Probing, save them to `./_noise/`.

# Experiments


## Step 1. Importance Probing (IP) to indentify important kernels for target adaptation

~~~bash
bash _bash_importance_probing.sh
~~~

We can obtain the estimated Fisher information of modulated kernels and it will be saved in `./_output_style_gan/args.exp/checkpoints/filter_fisher_g.pt` and `./_output_style_gan/args.exp/checkpoints/filter_fisher_d.pt`

## Step 2.  Adaptation-Aware Kernel Modulation (AdAM) for Few-shot Image Generation

~~~bash
# you can tune hyperparameters here
bash _bash_main_adaptation.sh
~~~

Training dynamics and evaluation results will be shown on [`wandb`](https://wandb.ai/site)

We note that, ideally Step 1. and Step 2. can be combined together. Here, for simplicity we use two steps as demonstration.

### Evaluation of Intra-LPIPS:
Use Babies and AFHQ-Cat as example: download images from [here](https://drive.google.com/file/d/1JQDEV_I2wIULqjIp6ms1hpsGgUYZKTgG/view?usp=share_link), then move the unzipped folder into `./cluster_center`, then refer to `Evaluator` in `AdAM_main_adaptation.py`.

# Data Repository
The estimated fisher information (i.e., the output of Importance Probing) and Weights (i.e., the output of the main adaptation corresponding to Figure 4 in the main paper) can be found [Here](https://drive.google.com/drive/folders/11uZjqJZl7ImapEndU4locAr2miHCJvDY?usp=share_link).


## Train your own GAN !

We provide all 10-shot target images and models used in our main paper and Supplementary. You can also adapt to other images selected by yourself.

Source GAN:
- [FFHQ](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view)
- [LSUN-Church](https://drive.google.com/file/d/18NlBBI8a61aGBHA1Tr06DQYlf-DRrBOH/view)
- [LSUN-Cars](https://drive.google.com/file/d/1O-yWYNvuMmirN8Q0Z4meYoSDtBfJEjGc/view)
- ...

Target Samples: [Link](https://drive.google.com/drive/folders/10skBzKjr8jJbWvTXKgA0yj-gT-aojRIE?usp=sharing)

- Babies
- Sunglasses
- MetFaces
- AFHQ-Cat
- AFHQ-Dog
- AFHQ-Wild
- Sketches
- Amedeo Modigliani's Paintings
- Rafael's Paintings
- Otto Dix's Paintings
- Haunted houses
- Van Gogh houses
- Wrecked cars
- ...

Follow the experiment part in this repo and you can produce your customized results.

# Bibtex
If you find this project useful in your research, please consider citing our paper:

```
@inproceedings{
zhao2022fewshot,
title={Few-shot Image Generation via Adaptation-Aware Kernel Modulation},
author={Yunqing Zhao and Keshigeyan Chandrasegaran and Milad Abdollahzadeh and Ngai-man Cheung},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=Z5SE9PiAO4t}
}
```

# Acknowledgement: 

We appreciate the wonderful base implementation of StyleGAN-V2 from [@rosinality](https://github.com/rosinality). We thank [@mseitzer](https://github.com/mseitzer/pytorch-fid), [@Ojha](https://github.com/utkarshojha/few-shot-gan-adaptation) and [@richzhang](https://github.com/richzhang/PerceptualSimilarity) for their implementations on FID score and intra-LPIPS.

We also thank for the useful training and evaluation tool used in this work, from [@Miaoyun](https://github.com/MiaoyunZhao/GANmemory_LifelongLearning).



