import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar
import csv
import datetime

from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminatorLarge
from gan.discriminator import ConditionalDiscriminatorSmall
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights


# Argument parser
parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes', 'all']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
parser.add_argument("--csv", action='store_true', help="Enable CSV logging")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])

# Function to get dataset
def get_dataset(name):
    if name == 'cityscapes':
        return Cityscapes(root='.', transform=transforms, download=True, mode='train'), \
               Cityscapes(root='.', transform=transforms, download=True, mode='val')
    elif name == 'maps':
        return Maps(root='.', transform=transforms, download=True, mode='train'), \
               Maps(root='.', transform=transforms, download=True, mode='val')
    else:  # Default to 'facades'
        return Facades(root='.', transform=transforms, download=True, mode='train'), \
               Facades(root='.', transform=transforms, download=True, mode='val')

# Datasets to process
datasets_to_process = ['facades', 'maps', 'cityscapes'] if args.dataset == 'all' else [args.dataset]

for dataset_name in datasets_to_process:
    print(f'Downloading and processing "{dataset_name.upper()}" dataset!')

    # Initialize models for each dataset
    generator = UnetGenerator().to(device)
    discriminatorL = ConditionalDiscriminatorLarge().to(device)
    discriminatorS = ConditionalDiscriminatorSmall().to(device)

    # Initialize optimizers for each dataset
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizerL = torch.optim.Adam(discriminatorL.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizerS = torch.optim.Adam(discriminatorS.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Initialize loss functions
    g_criterion = GeneratorLoss(alpha=100)
    d_criterion = DiscriminatorLoss()

    # Load dataset
    train_dataset, val_dataset = get_dataset(dataset_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    csv_file = None
    csv_writer = None
    if args.csv:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = open(f'{dataset_name}_{timestamp}_training_log.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', ' G Training Loss', 'D Training Loss', 'G Validation Loss', 'D Validation Loss', 'Epoch Time (s)'])


    print('Start of training process for', dataset_name, 'dataset!')
    logger = Logger(filename=dataset_name)
    
    # Training loop
    for epoch in range(args.epochs):
        generator.train()
        discriminatorL.train()
        discriminatorS.train()
        ge_loss=0.0
        de_loss_large = 0.0  # Large discriminator loss
        de_loss_small = 0.0  # Small discriminator loss
        start = time.time()
        bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(train_dataloader))
        for x, real in train_dataloader:
            x = x.to(device)
            real = real.to(device)

            # Generator`s loss
            fake = generator(x)
            fake_pred_large = discriminatorL(fake, x)
            fake_pred_small = discriminatorS(fake, x)
            g_loss_large = g_criterion(fake, real, fake_pred_large)
            g_loss_small = g_criterion(fake, real, fake_pred_small)
            g_loss = (g_loss_large + g_loss_small) / 2

            # Discriminator`s loss
            fake = generator(x).detach()
            fake_pred_large = discriminatorL(fake, x)
            fake_pred_small = discriminatorS(fake, x)
            real_pred_large = discriminatorL(real, x)
            real_pred_small = discriminatorS(real, x)
            d_loss_large = d_criterion(fake_pred_large, real_pred_large)
            d_loss_small = d_criterion(fake_pred_small, real_pred_small)

            # Generator`s params update
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Discriminator`s params update
            d_optimizerL.zero_grad()
            d_optimizerS.zero_grad()
            d_loss_large.backward()
            d_loss_small.backward()
            d_optimizerL.step()
            d_optimizerS.step()
            # add batch losses
            ge_loss += g_loss.item()
            de_loss_large += d_loss_large.item()
            de_loss_small += d_loss_small.item()
            bar.next()
        bar.finish()

        # Validation Step
        generator.eval()
        discriminator.eval()
        ge_val_loss=0.
        de_val_loss=0.
        with torch.no_grad():
            bar = IncrementalBar(f'[Validation]', max=len(val_dataloader))
            for val_x, val_real in val_dataloader:
                val_x = val_x.to(device)
                val_real = val_real.to(device)

                # Generator`s loss for validation
                val_fake = generator(val_x)
                val_fake_pred = discriminator(val_fake, val_x)
                val_g_loss = g_criterion(val_fake, val_real, val_fake_pred)

                # Discriminator`s loss for validation
                val_fake = generator(val_x).detach()
                val_fake_pred = discriminator(val_fake, val_x)
                val_real_pred = discriminator(val_real, val_x)
                val_d_loss = d_criterion(val_fake_pred, val_real_pred)

                # add batch losses
                ge_val_loss += val_g_loss.item()
                de_val_loss += val_d_loss.item()
                bar.next()
            bar.finish()

        # obtain per epoch losses for both training and validation
        g_loss = ge_loss / len(train_dataloader)
        d_loss = de_loss / len(train_dataloader)
        val_g_loss = ge_val_loss / len(val_dataloader)
        val_d_loss = de_val_loss / len(val_dataloader)

        # count timeframe
        end = time.time()
        tm = (end - start)

        logger.add_scalar('generator_loss', g_loss, epoch+1)
        logger.add_scalar('discriminator_loss', d_loss, epoch+1)
        logger.add_scalar('val_generator_loss', val_g_loss, epoch + 1)
        logger.add_scalar('val_discriminator_loss', val_d_loss, epoch + 1)

        logger.save_weights(generator.state_dict(), 'generator-base')
        logger.save_weights(discriminator.state_dict(), 'discriminator-base')
        if args.csv:
            csv_writer.writerow([epoch + 1,g_loss, d_loss, val_g_loss, val_d_loss, tm])
        print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] [Val G loss: %.3f] [Val D loss: %.3f] ETA: %.3fs"
            % (epoch + 1, args.epochs, g_loss, d_loss, val_g_loss, val_d_loss, tm))


    logger.close()
    print('End of training process for', dataset_name, 'dataset!')

print('All datasets processed!')