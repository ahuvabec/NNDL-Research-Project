import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss() # to calculate L1 (mean absolute error) between the generated(fake) image and the target image
        
    def forward(self, fake, real, fake_pred):
        # takes 3 inputs: generated(fake) images, real images, and generated(fake) images predictions
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha* self.l1(fake, real)
        return loss
    
    
class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        # takes 2 inputs: generated(fake) images and real images predictions
        fake_target = torch.zeros_like(fake_pred) # zeros because these are the predictions for the fake generated images
        real_target = torch.ones_like(real_pred) # ones because these are the predictions for the real images
        fake_loss = self.loss_fn(fake_pred, fake_target) # cross entropy loss between generated images predictions and zeros
        real_loss = self.loss_fn(real_pred, real_target) # cross entropy loss between real images predictions and ones
        loss = (fake_loss + real_loss)/2
        return loss
