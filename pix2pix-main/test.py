import torch
from torch.utils.data import DataLoader
from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
import argparse
from dataset import Cityscapes, Facades, Maps
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
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

num_plots = 2

# Generate outputs using the trained generator
with torch.no_grad():
    bar = IncrementalBar(f'[Testing ..]', max=len(test_dataloader))
    for i, (x, real) in enumerate(test_dataloader):
        x = x.to(device)
        real = real.to(device)
        # print("x.size(): ", x.size())
        # print("real.size(): ", real.size())

        # Generate output
        output = generator(x)
        # print("output.size(): ", output.size())

        # # discriminator (?)
        # fake_pred = discriminator(output, x)
        # print("fake_pred.size(): ", fake_pred.size())

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

            plt.show()
        bar.next()
    bar.finish()
print("Testing process completed.")
