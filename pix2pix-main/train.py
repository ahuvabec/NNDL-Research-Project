import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar

from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights

parser = argparse.ArgumentParser(prog = 'top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])
# models
print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

# loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

# datasets
print(f'Downloading "{args.dataset.upper()}" dataset!')
if args.dataset == 'cityscapes':
    train_dataset = Cityscapes(root='.', transform=transforms, download=True, mode='train')
    val_dataset = Cityscapes(root='.', transform=transforms, download=True, mode='val')
elif args.dataset == 'maps':
    train_dataset = Maps(root='.', transform=transforms, download=True, mode='train')
    val_dataset = Maps(root='.', transform=transforms, download=True, mode='val')
else:
    train_dataset = Facades(root='.', transform=transforms, download=True, mode='train')
    val_dataset = Facades(root='.', transform=transforms, download=True, mode='val')

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print('Start of training process!')
logger = Logger(filename=args.dataset)
for epoch in range(args.epochs):
    generator.train()
    discriminator.train()
    ge_loss=0.
    de_loss=0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(train_dataloader))
    for x, real in train_dataloader:
        x = x.to(device)
        real = real.to(device)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # add batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()
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
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] [Val G loss: %.3f] [Val D loss: %.3f] ETA: %.3fs"
          % (epoch + 1, args.epochs, g_loss, d_loss, val_g_loss, val_d_loss, tm))
logger.close()
print('End of training process!')
