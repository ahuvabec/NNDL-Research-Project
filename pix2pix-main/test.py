import torch
from torch.utils.data import DataLoader
from dataset import Cityscapes, Facades, Maps
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
import argparse
from dataset import Cityscapes, Facades, Maps
import matplotlib.pyplot as plt
#from torchvision.transforms.functional import to_pil_image

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

# Define the testing dataset and dataloader
transforms = T.Compose([T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])

# Choose the appropriate dataset based on the training dataset
# datasets
print(f'Accessing "{args.dataset.upper()}" dataset!')
if args.dataset == 'cityscapes':
    test_dataset = Cityscapes(root='.', transform=transforms, download=False, mode='test')
elif args.dataset == 'maps':
    test_dataset = Maps(root='.', transform=transforms, download=False, mode='test')
else:
    test_dataset = Facades(root='.', transform=transforms, download=False, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Generate outputs using the trained generator
with torch.no_grad():
    for i, (input_data, _) in enumerate(test_dataloader):
        input_data = input_data.to(device)
        print("input_data.size(): ", input_data.size())

        # Generate output
        output = generator(input_data)
        print("output.size(): ", output.size())

        # # Use the discriminator to classify the generated output
        # fake_pred = discriminator(output, input_data)
        # print("fake_pred.size(): ", fake_pred.size())

        # Save or visualize the generated output and discriminator prediction as needed
        # For example, save the output image and discriminator prediction
        output_image = output.squeeze().cpu().numpy().transpose((1, 2, 0))
        print("output_image.size(): ", output.size())
        # Save or visualize the output_image and fake_pred as needed
        # ...

        # # Convert the NumPy array to a PIL Image for visualization
        # output_pil = to_pil_image(torch.from_numpy((output_image + 1) / 2).permute(2, 0, 1))  # Assuming output is in range [-1, 1]

        # Display the generated image
        plt.imshow(output_image[0])
        # plt.title(f"Generated Output - Discriminator Prediction: {fake_pred[0].item()}")
        plt.title("Generated Output")
        plt.show()

        print(f"Generated output for sample {i + 1} - Discriminator Prediction: {fake_pred[0].item()}")

print("Testing process completed.")
