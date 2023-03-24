<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Exploring Incompatible Knowledge Transfer<br>in Few-shot Image Generation</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://scholar.google.com/citations?user=kQA0x9UAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Yunqing Zhao<sup>1</sup></a>&nbsp;/&nbsp;
    <a href="https://duchao0726.github.io/" target="_blank" style="text-decoration: none;">Chao Du<sup>2</sup></a>&nbsp;/&nbsp;
    <a href="https://miladabd.github.io/" target="_blank" style="text-decoration: none;">Milad Abdollahzadeh<sup>1</sup></a>&nbsp;/&nbsp;
    <a href="https://p2333.github.io/" target="_blank" style="text-decoration: none;">Tianyu Pang<sup>2</sup></a></br>
    <a href="https://linmin.me/" target="_blank" style="text-decoration: none;">Min Lin<sup>2</sup></a>&nbsp;/&nbsp;
    <a href="https://yanshuicheng.ai/" target="_blank" style="text-decoration: none;">Shuicheng Yan<sup>2</sup></a>&nbsp;/&nbsp;
    <a href="https://sites.google.com/site/mancheung0407/" target="_blank" style="text-decoration: none;">Ngai&#8209;Man Cheung<sup>1</sup></a></br>
<sup>1</sup>Singapore University of Technology and Design (SUTD)&emsp;&emsp; <sup>2</sup>Sea AI Lab<br/>
</p>

<p align='center';>
<b>
<em>The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023;</em> <br>
<em>Vancouver Convention Center, Vancouver, British Columbia, Canada.</em>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://github.com/yunqing-me/RICK" target="_blank" style="text-decoration: none;">Project Page</a>&nbsp;/&nbsp;
    <a href="https://github.com/yunqing-me/RICK" target="_blank" style="text-decoration: none;">Poster</a>&nbsp;/&nbsp;
    <a href="https://github.com/yunqing-me/RICK" target="_blank" style="text-decoration: none;">Slides</a>&nbsp;/&nbsp;
    <a href="https://github.com/yunqing-me/RICK" target="_blank" style="text-decoration: none;">Paper</a>&nbsp;
    <!-- /&nbsp; -->
    <!-- <a href="https://recorder-v3.slideslive.com/?share=74947&s=c88e53c5-a3c2-46c9-9719-092b74eca0c2" target="_blank" style="text-decoration: none;">Talk</a>&nbsp; -->
</b>
</p>


<!-- ---------------------------------------------------------------------- -->

<!-- #### TL, DR: 
```
In this research, we propose Adaptation-Aware Kernel Modulation (AdAM) for few-shot image generation, that aims to identify kernels in source GAN important for target adaptation. 

The model can perform GAN adaptation using very few samples from target domains with different proximity to the source.
``` -->
Code/Project Page will be actively updated

# Installation and Environment:
- Platform: Linux
- NVIDIA A100 PCIe 40GB, CuDNN 11.4
- lmdb, tqdm, wandb

A suitable conda environment named `fsig` can be created and activated with:

```
conda env create -f environment.yaml -n fsig
conda activate fsig
```


# Prepare the datasets

## Step 1. 
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

## Step 2. 
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

## Step 3. 
Download the GAN model pretrained on FFHQ from [here](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view). Then, save it to `./_pretrained/style_gan_source_ffhq.pt`.


# Bibtex
If you find this project useful in your research, please consider citing our paper:

```
@InProceedings{zhao2023fsig,
   author    = {Zhao, Yunqing and Du, Chao and Abdollahzadeh, Milad and Pang, Tianyu and Lin, Min and YAN, Shuicheng and Cheung, Ngai-Man},
    title     = {Exploring Incompatible Knowledge Transfer in Few-shot Image Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
}
```

# Acknowledgement: 

We appreciate the wonderful base implementation of StyleGAN-V2 from [@rosinality](https://github.com/rosinality). We thank [@mseitzer](https://github.com/mseitzer/pytorch-fid), [@Ojha](https://github.com/utkarshojha/few-shot-gan-adaptation) and [@richzhang](https://github.com/richzhang/PerceptualSimilarity) for their implementations on FID score and intra-LPIPS.

We also thank for the useful training and evaluation tool used in this work, from [@Miaoyun](https://github.com/MiaoyunZhao/GANmemory_LifelongLearning).



