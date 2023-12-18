# NNDL-Research-Project

This project is a collaborative effort developed as the final group project for the COMS W4995: Neural Networks Deep Learning course in Fall 2023, taught by Richard Zemel.

Authors: Maitar Asher, Ahuva Bechhofer, Shmuel Berman

# About

Building upon Isola et al.'s pioneering work in "Image-to-Image Translation with Conditional Adversarial Networks", this project focuses on the Pix2Pix software associated with their [paper](https://arxiv.org/pdf/1611.07004.pdf). By exploring modifications and refinements to their CGAN network architecture, including variations in the generator's and discriminator's design, our objective is to delve deeper into the underlying mechanisms contributing to the remarkable efficacy of Pix2Pix.

# Acknowledgments:
We utilized the codebase from https://github.com/akanametov/pix2pix as a foundation and incorporated our custom modifications.

# Install
```
git clone https://github.com/ahuvabec/NNDL-Research-Project.git
cd pix2pix-main
```

# Training

## Base Model

```
python train.py [--epochs EPOCHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--lr LR]
```
```
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs
  --dataset DATASET     Name of the dataset: ['facades', 'maps', 'cityscapes']
  --batch_size BATCH_SIZE
                        Size of the batches
  --lr LR               Adams learning rate
```

## Multi-scale Discriminator
 ```
 additional arguments:
   --ld_alpha            Alpha weight for large discriminator
 ```

### Consistent Weights Approach

```
python train2d.py [--epochs EPOCHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--lr LR] [--ld_alpha LD_ALPHA]
```

### Dynamic Updates Approach

#### V1

```
python train2d_dynamic.py [--epochs EPOCHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--lr LR] [--ld_alpha LD_ALPHA]
```

#### V2

```
python train2d_dynamic_v2.py [--epochs EPOCHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--lr LR] [--ld_alpha LD_ALPHA]
```
## Skip Connection Variation
# Training
```
python train_skip.py [--epochs EPOCHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--lr LR] [--gan_folder {gan,gan_multiply,gan_add}]
```
```
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs
  --dataset DATASET     Name of the dataset: ['facades', 'maps', 'cityscapes']
  --batch_size BATCH_SIZE
                        Size of the batches
  --lr LR               Adams learning rate
  --gan_folder {gan,gan_multiply,gan_add}
                        Folder to import GAN from
```
# Testing

```
python test.py [--generator_path GENERATOR_PATH] [--dataset DATASET] [--num_imgs NUM_IMGS] [--gan_folder {gan,gan_multiply,gan_add}]
```
```
optional arguments:
  -h, --help            show this help message and exit
  --generator_path GENERATOR_PATH
                        Path to the saved generator weights
  --dataset DATASET     Name of the dataset: ['facades', 'maps', 'cityscapes']
  --num_imgs NUM_IMGS   Number of images to generate
  --gan_folder {gan,gan_multiply,gan_add}
                        Folder to import GAN from 

```
