import torch
from torch.utils.data import DataLoader
from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
import argparse
from dataset import Cityscapes, Facades, Maps
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
import os
from gan.utils import Logger
#from PIL import Image


# Argument Parser
parser = argparse.ArgumentParser(prog='top', description='Test Pix2Pix Generator and Discriminator')
parser.add_argument("--generator_path", type=str, required=True, help="Path to the saved generator weights")
parser.add_argument("--discriminator_path", type=str, required=True, help="Path to the saved discriminator weights")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
args = parser.parse_args()

# Define the device
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the trained generator and discriminator
generator = UnetGenerator().to(device)
generator.load_state_dict(torch.load(args.generator_path))
generator.eval()

discriminator = ConditionalDiscriminator().to(device)
discriminator.load_state_dict(torch.load(args.discriminator_path))
discriminator.eval()

# Initialize loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

# Original transformation
transforms = T.Compose([T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])
# Inverse transformation
inverse_transform = T.Compose([
    T.ToImage(),
])

# Choose the appropriate dataset based on the training dataset (should be downloaded already)
# datasets
print(f'Accessing "{args.dataset.upper()}" dataset!')
if args.dataset == 'cityscapes':
    test_dataset = Cityscapes(root='.', transform=transforms, download=False, mode='test')
elif args.dataset == 'maps':
    test_dataset = Maps(root='.', transform=transforms, download=False, mode='test')
else:
    test_dataset = Facades(root='.', transform=transforms, download=False, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


logger = Logger(filename=args.dataset+'_test')

num_plots = 2
ge_loss=0.
de_loss=0.

# Generate outputs and using the trained generator and calculate loss
with torch.no_grad():
    bar = IncrementalBar(f'[Testing ..]', max=len(test_dataloader))
    for i, (x, real) in enumerate(test_dataloader):
        x = x.to(device)
        real = real.to(device)

        # Generate output & loss
        output = generator(x)
        fake_pred = discriminator(output, x)
        g_loss = g_criterion(output, real, fake_pred)

        # Discriminator`s loss
        output_d = generator(x).detach()
        fake_pred = discriminator(output_d, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        ge_loss += g_loss.item()
        de_loss += d_loss.item()

        # Plot
        if (i < num_plots):
            # Convert tensors back to PIL images
            input_image = inverse_transform(x.squeeze().cpu())
            generated_image = inverse_transform(output.squeeze().cpu())
            real_image = inverse_transform(real.squeeze().cpu())

            # Display the images
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title("Input Image")

            plt.subplot(1, 3, 2)
            plt.imshow(generated_image)
            plt.title("Generated Image")

            plt.subplot(1, 3, 3)
            plt.imshow(real_image)
            plt.title("Real Image")

            # Save the images
            save_dir = 'runs/saved_images/' + args.dataset
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'image_{i}.png'))

            plt.show()
        bar.next()
    bar.finish()

g_loss = ge_loss / len(test_dataloader)
d_loss = de_loss / len(test_dataloader)
logger.add_scalar('generator_loss', g_loss, 1)
logger.add_scalar('discriminator_loss', d_loss, 1)
logger.close()
print("[G loss: %.3f] [D loss: %.3f]"
            % (g_loss, d_loss))
print("Testing process completed.")
