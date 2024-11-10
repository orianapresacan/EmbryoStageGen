# Embryo Image Generation and Classification

## [Paper](TBD) | [GitHub](https://github.com/orianapresacan/Embryo)

<img src="./images/embryo_synthetic.jpg" width="750" height="150"/> 

## Overview

This repository contains code for training two generative models—a diffusion model (LDM) and a generative adversarial network (StyleGAN2)—to produce synthetic images of embryos at various developmental stages: 2-cell, 4-cell, 8-cell, blastocyst, and morula. It also includes scripts for training and testing three classification models—ResNet, VGG, and Vision Transformer—used to classify these embryos.

## Repository Structure

- `Classification/` - Scripts for training ResNet, VGG, and ViT for cell classification, along with the models' checkpoints.
  
- `latent-diffusion/` - Code for generating synthetic data using the Latent Diffusion Model. This code was sourced from its original repository, which can be accessed [here](https://github.com/CompVis/latent-diffusion).

  - Add the checkpoints for the vq-f4 autoencoder model to `models/first_stage_models/vq-f4/` directory. The checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/11FvUtObPUTdT-lUb8H_y3uOFer3SDqM4?usp=sharing).
    
  - Use the following command for training:
  ```bash
  python main.py --base configs/latent-diffusion/embryo-ldm-vq-4.yaml -t --gpus 0
  ```
  
  - Use the following command for sampling:
  ```bash
  python sample_diffusion.py --resume models/ldm/embryo/[checkpoint-file] --n_samples 10
  ```
The checkpoints for our trained latent diffusion models, corresponding to each cell stage (2-cell, 4-cell, 8-cell, Morula, Blastocyst) can be downloaded from [this link](https://drive.google.com/drive/folders/19EnY1wUUA0ZQVfI5E3J6IAeAECmcFK3y?usp=drive_link).

- `StyleGAN/` - Code for generating synthetic data using StyleGAN. The code is based on the repository available [here]().

## Synthetic Dataset

The embryo synthetic dataset containing images generated using the LDM and StyleGAN2 models can be found [here](https://drive.google.com/file/d/1egpag71fUtZTcB04Bn4mLeVo5s2jh9-W/view?usp=drive_link). These images were used alongside the real dataset to train the classification models.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## For more details:
Please contact: orianapresacan@gmail.com
