import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar
import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan2d.discriminator2d import ConditionalDiscriminatorLarge
from gan2d.discriminator2d import ConditionalDiscriminatorSmall
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights


# Argument parser
parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes', 'all']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
parser.add_argument("--csv", action='store_true', help="Enable CSV logging")
parser.add_argument("--ld_alpha", type=float, default=0.5, help="Alpha weight for large discriminator")
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
        csv_writer.writerow(['Epoch', ' G Training Loss', 'D L Training Loss', 'D S Training Loss', 'Epoch Time (s)'])


    print('Start of training process for', dataset_name, 'dataset!')
    logger = Logger(filename=f'gan2d_dynamic_{dataset_name}_{args.epochs}_epochs_{args.batch_size}_bs_{args.lr}_lr_{args.ld_alpha}_ld_alpha')

    # Lists to store losses for plotting
    train_generator_losses = []
    train_discriminator_large_losses = []
    train_discriminator_small_losses = []

    #variable for dynamic alpha
    dynamic = 0

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

        if epoch % 10 == 0:
            dynamic += 1

        for x, real in train_dataloader:
            x = x.to(device)
            real = real.to(device)

            # Generator`s loss
            fake = generator(x)
            fake_pred_large = discriminatorL(fake, x)
            fake_pred_small = discriminatorS(fake, x)
            g_loss_large = g_criterion(fake, real, fake_pred_large)
            g_loss_small = g_criterion(fake, real, fake_pred_small)
            # g_loss = (args.ld_alpha)**(dynamic)*(g_loss_large) + (1-args.ld_alpha)**(dynamic)*g_loss_small
            g_loss = (args.ld_alpha)**(dynamic)*(g_loss_large) + (1-args.ld_alpha**(dynamic))*g_loss_small
            if epoch % 10 == 0:
                print(f'old: weight_small={(args.ld_alpha)**(dynamic)} weight_large={(1-args.ld_alpha)**(dynamic)}')
                print(f'new: weight_small={(args.ld_alpha)**(dynamic)} weight_large={(1-args.ld_alpha**(dynamic))}')

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

        # Ignoring Val for now
        # # Validation Step
        # generator.eval()
        # discriminator.eval()
        # ge_val_loss=0.
        # de_val_loss=0.
        # with torch.no_grad():
        #     bar = IncrementalBar(f'[Validation]', max=len(val_dataloader))
        #     for val_x, val_real in val_dataloader:
        #         val_x = val_x.to(device)
        #         val_real = val_real.to(device)
        #
        #         # Generator`s loss for validation
        #         val_fake = generator(val_x)
        #         val_fake_pred = discriminator(val_fake, val_x)
        #         val_g_loss = g_criterion(val_fake, val_real, val_fake_pred)
        #
        #         # Discriminator`s loss for validation
        #         val_fake = generator(val_x).detach()
        #         val_fake_pred = discriminator(val_fake, val_x)
        #         val_real_pred = discriminator(val_real, val_x)
        #         val_d_loss = d_criterion(val_fake_pred, val_real_pred)
        #
        #         # add batch losses
        #         ge_val_loss += val_g_loss.item()
        #         de_val_loss += val_d_loss.item()
        #         bar.next()
        #     bar.finish()

        # obtain per epoch losses for both training and validation
        g_loss = ge_loss / len(train_dataloader)
        d_loss_large = de_loss_large / len(train_dataloader)
        d_loss_small = de_loss_small / len(train_dataloader)
        # val_g_loss = ge_val_loss / len(val_dataloader)
        # val_d_loss = de_val_loss / len(val_dataloader)

        # count timeframe
        end = time.time()
        tm = (end - start)

        # Append losses to the lists
        train_generator_losses.append(g_loss)
        train_discriminator_large_losses.append(d_loss_large)
        train_discriminator_small_losses.append(d_loss_small)

        logger.add_scalar('generator_loss', g_loss, epoch+1)
        logger.add_scalar('discriminator_large_loss', d_loss_large, epoch+1)
        logger.add_scalar('discriminator_small_loss', d_loss_small, epoch+1)
        # logger.add_scalar('val_generator_loss', val_g_loss, epoch + 1)
        # logger.add_scalar('val_discriminator_loss', val_d_loss, epoch + 1)

        # Save trained models
        logger.save_weights(generator.state_dict(), f'gan2d_dynamic_{dataset_name}_{args.epochs}_epochs_{args.batch_size}_bs_{args.lr}_lr_{args.ld_alpha}_ld_alpha_generator')
        logger.save_weights(discriminatorL.state_dict(), f'gan2d_dynamic_{dataset_name}_{args.epochs}_epochs_{args.batch_size}_bs_{args.lr}_lr_discriminatorL')
        logger.save_weights(discriminatorS.state_dict(), f'gan2d_dynamic_{dataset_name}_{args.epochs}_epochs_{args.batch_size}_bs_{args.lr}_lr_discriminatorS')

        if args.csv:
            csv_writer.writerow([epoch + 1,g_loss, d_loss_large, d_loss_small, tm])
        print("[Epoch %d/%d] [G loss: %.3f] [D loss large: %.3f] [D loss small: %.3f] ETA: %.3fs"
            % (epoch + 1, args.epochs, g_loss, d_loss_large, d_loss_small, tm))

    logger.close()

    # Plot the training losses
    epochs_range = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Discriminator Loss on the left y-axis
    ax1.plot(epochs_range, train_discriminator_large_losses, label='Large Discriminator Loss')
    ax1.plot(epochs_range, train_discriminator_small_losses, label='Small Discriminator Loss', linestyle='dashed', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Discriminator Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a secondary y-axis for Generator Loss on the right
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, train_generator_losses, label='Training Generator Loss', color='green')
    ax2.set_ylabel('Generator Loss', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    #ax2.legend(loc='upper right')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # Set x-axis ticks as integer values
    plt.title(f'GAN2d dynamic {dataset_name[0].upper()}{dataset_name[1:]} Train Losses ({args.batch_size} bs, {args.lr} lr, {args.ld_alpha} ld_alpha)')
    plt.legend()

    # Save the plot to a file
    save_dir = 'runs/plots/' + dataset_name
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir,f'gan2d_dynamic_{args.epochs}_epochs_{args.batch_size}_bs_{args.lr}_lr_{args.ld_alpha}_ld_alpha_train_plot.png'))
    plt.show()

    print('End of training process for', dataset_name, 'dataset!')

print('All datasets processed!')
