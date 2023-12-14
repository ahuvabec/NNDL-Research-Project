import torch
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
import os
from PIL import Image

from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from dataset import Cityscapes, Facades, Maps
from gan.utils import Logger

# Argument Parser
parser = argparse.ArgumentParser(prog='top', description='Test Pix2Pix Generator and Discriminator')
parser.add_argument("--generator_path", type=str, required=True, help="Path to the saved generator weights")
#parser.add_argument("--discriminator_path", type=str, required=True, help="Path to the saved discriminator weights")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--num_imgs", type=int, default="1", help="Num of images to generate (max is the number of imgs in data set)")
args = parser.parse_args()

# Define the device
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the trained generator and discriminator
generator = UnetGenerator().to(device)
generator.load_state_dict(torch.load(args.generator_path))
generator.eval()

# discriminator = ConditionalDiscriminator().to(device)
# discriminator.load_state_dict(torch.load(args.discriminator_path))
# discriminator.eval()

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
if args.dataset == 'cityscapes': # dataset have train and val  only
    test_dataset = Cityscapes(root='.', transform=transforms, download=False, mode='val')
elif args.dataset == 'maps': # dataset have train and val  only
    test_dataset = Maps(root='.', transform=transforms, download=False, mode='val')
else:
    test_dataset = Facades(root='.', transform=transforms, download=False, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


logger = Logger(filename=args.dataset+'_test')

num_imgs = min(args.num_imgs, len(test_dataloader))
num_show = 2
# ge_loss=0.
# de_loss=0.

# Generate outputs and using the trained generator and calculate loss
with torch.no_grad():
    bar = IncrementalBar(f'[Testing ..]', max=len(test_dataloader))
    for i, (x, real) in enumerate(test_dataloader):
        x = x.to(device)
        real = real.to(device)

        # Generate output & loss
        output = generator(x)
        # fake_pred = discriminator(output, x)
        # g_loss = g_criterion(output, real, fake_pred)

        # # Discriminator`s loss
        # output_d = generator(x).detach()
        # fake_pred = discriminator(output_d, x)
        # real_pred = discriminator(real, x)
        # d_loss = d_criterion(fake_pred, real_pred)
        #
        # ge_loss += g_loss.item()
        # de_loss += d_loss.item()

        # Plot
        if (i < num_imgs):

            # Convert tensors back to images
            # (1) Transpose from (3,w,h) to (w,h,w)
            # (2) Generator outputs images with pixel values in the range [-1, 1].
            # The +1 operation is a part of the process to bring these values to the non-negative range [0, 2]
            # (3) The / 2 operation is a scaling factor that brings the pixel values into a standard image range of [0, 1].
            input_image = (x[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2
            generated_image = (output[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2
            real_image = (real[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2

            # Display the images
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title("Input Image")

            plt.subplot(1, 3, 2)
            plt.imshow(real_image)
            plt.title("Ground Truth")

            plt.subplot(1, 3, 3)
            plt.imshow(generated_image)
            plt.title("Generated Image")


            # Save the images in model params directories
            save_dir = 'runs/saved_images/'
            # Find the starting index of the dataset name
            start_index = args.generator_path.find(args.dataset)
            if (start_index != -1):
                start_index += len(args.dataset) + 1
                if (args.generator_path.find('gan2d_dynamic') != -1):
                    end_index = args.generator_path.find('_ld_alpha', start_index)
                    model_params = 'gan2d_dynamic_' + args.generator_path[start_index:end_index + 9]
                elif (args.generator_path.find('gan2d') != -1):
                    end_index = args.generator_path.find('_ld_alpha', start_index)
                    model_params = 'gan2d_' + args.generator_path[start_index:end_index + 9]
                else:
                    end_index = args.generator_path.find('_lr', start_index)
                    model_params = args.generator_path[start_index:end_index + 3]
                save_dir = save_dir + model_params
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{args.dataset}_image_{i+1}.png'))

            if i < num_show:
                plt.show()
            plt.close()

            # Save singular imags for calculating FID scores

            # Create Image objects from NumPy arrays
            input_image = Image.fromarray((input_image * 255).astype('uint8'))
            generated_image = Image.fromarray((generated_image * 255).astype('uint8'))
            real_image = Image.fromarray((real_image * 255).astype('uint8'))

            save_dir_input = os.path.join(save_dir, "input")
            save_dir_generated = os.path.join(save_dir, "generated")
            save_dir_real = os.path.join(save_dir, "real")

            os.makedirs(save_dir_input, exist_ok=True)
            os.makedirs(save_dir_generated, exist_ok=True)
            os.makedirs(save_dir_real, exist_ok=True)

            # Save the images to files
            input_image.save(os.path.join(save_dir + "/input", f'{args.dataset}_image_{i+1}.png'))
            generated_image.save(os.path.join(save_dir + "/generated", f'{args.dataset}_image_{i+1}.png'))
            real_image.save(os.path.join(save_dir + "/real", f'{args.dataset}_image_{i+1}.png'))

        bar.next()
    bar.finish()

# g_loss = ge_loss / len(test_dataloader)
# d_loss = de_loss / len(test_dataloader)
# logger.add_scalar('generator_loss', g_loss, 1)
# logger.add_scalar('discriminator_loss', d_loss, 1)
# logger.close()
# print("[G loss: %.3f] [D loss: %.3f]"
#             % (g_loss, d_loss))
print("Testing process completed.")
