import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch import einsum

import math
import numpy as np
from einops import rearrange
import os
import wandb
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Reproducbility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Environment variables
os.environ["WANDB_API_KEY"] = "44b27736c17cd4318936992ac992d9d787f1a5e5"

def load_data(config):
    """ Function: Load dataset partitions and apply transformations
        Input:    Batch size
        Output:   Train, val and test loader
    """

    # Data transformations
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: (2. * x) - 1.)  # rescale to [-1, 1]
    ])

    # Initialize dataset
    temp_data = MNIST(root='data', train=True, download=True, transform=transforms)
    train_data, val_data = random_split(temp_data, [55000,5000], generator=torch.Generator().manual_seed(42))
    test_data = MNIST(root='data', train=False, download=True, transform=transforms)

    # Initialize dataloaders
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, config.batch_size, shuffle=False)

    # Plot a few observations from train_loader
    x, _ = next(iter(train_loader))
    x = x.view(-1, 28, 28).numpy()
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(10):
        ax[i].imshow(x[i], cmap='gray')
        ax[i].axis('off')
    plt.show()

    return train_loader, val_loader, test_loader

class PositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Desire dimension of positional embeddings for each time step
        self.dim = dim

    def forward(self, time):
        """ Function: Returns transformer-style position embedding of time step (noise level) in batch
            Input:    Tensor shape (batch_size,1) with the time steps t of each image in batch
            Output:   Tensor shape (batch_size,dim) where dim is the dimension of position embeddings
        """

        # Transfer time indexes to correct device (GPU or CPU)
        device = time.device

        # Calculate embeeding using non-learned formula (from transformer RP)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups):
        super().__init__()

        self.groups = groups

        # Define layer, normalization and activation function
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=(3,3), padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        # Convolutional layer
        x = self.proj(x)

        if self.groups == 1:
            print("lol")

        # Group norm (normalize using mean and std for all pixels in groups
        # of feature maps [channels] for a single observation [not across batch])
        x = self.norm(x)

        # Activation function
        x = self.act(x)

        return x

class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_size=None):
        super().__init__()

        # MLP to additionally change time embedding besides the one applied in
        # the UNet class after creating the positional embedding. That one ended
        # with a linear layer with no activation - explaining why it makes sense
        # here to start with an activation function). The linear projection is
        # applied only to get to the correct dimension (to match dimesion of main
        # signal)
        if time_size is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_size, dim_out))

        # Define 2 blocks (see corresponding class)
        self.block1 = Block(dim_in, dim_out, groups=4)   # debug: 8
        self.block2 = Block(dim_out, dim_out, groups=4)  # debug: 8

        # This is the layer that will either do an identity mapping of the residual (most cases)
        # or do a linear projection (notice it is a 1x1 convolution) along the channel dimension
        # to increase the number of channels of the residual to the output of the computational
        # path in the skip connection (see ResNet architecture in fig. 3 in my ResNet notes)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_embs=None):
        """ Function: Apply 2 convolutions and inbetween add (condition on) time step embedding and add
                      this to the residual """

        h = self.block1(x)

        if time_embs is not None:
            time_embs = self.mlp(time_embs)

            # Add empty dimensions to time embedding tensor to match shape of main signal
            # - meaning shape (batch_size,channels) --> (batch_size,channels,H,W)
            h = rearrange(time_embs, "b c -> b c 1 1") + h

        h = self.block2(h)

        return h + self.res_conv(x)

class Attention(nn.Module):
    """ Multiheaded self-attention block used in UNet """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """ Linear attention block used in UNet """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class encoder_block(nn.Module):
    def __init__(self, dim_in, dim_out, time_size, is_last):
        super().__init__()

        self.res_block1 = ResNetBlock(dim_in, dim_out, time_size)
        self.res_block2 = ResNetBlock(dim_out, dim_out, time_size)
        self.group_norm = nn.GroupNorm(1, dim_out)  # TODO: Why only a single group - not e.g. 8?
        self.lin_attention = LinearAttention(dim_out)

        if not is_last:
            self.final = nn.Conv2d(dim_out, dim_out, 4, 2, 1)
        else:
            self.final = nn.Identity()

    def forward(self, x, time_embs):
        # Apply ResNet blocks and save result as x1
        x1 = self.res_block1(x,time_embs)
        x1 = self.res_block2(x1,time_embs)

        # Apply group norm and lin_attention block on x
        x2 = self.group_norm(x1)
        x2 = self.lin_attention(x2)

        # Add x1 and x2 (skip connection within encoder block)
        x3 = x1 + x2

        # Down-scale resolution unless this is the last encoder block # TODO: Why this unless thing?
        x4 = self.final(x3)

        # x3 is the residual for the corresponding decoder block and
        # x4 is input for the next encoder block (or bottleneck)
        return x4, x3

class bottleneck_block(nn.Module):
    def __init__(self, mid_dim, time_size):
        super().__init__()

        self.res_block1 = ResNetBlock(mid_dim, mid_dim, time_size)
        self.group_norm = nn.GroupNorm(1, mid_dim)  # TODO: Why only a single group - not e.g. 8?
        self.attention = Attention(mid_dim)
        self.res_block2 = ResNetBlock(mid_dim, mid_dim, time_size)

    def forward(self, x, time_embs):
        # Notice that the group norm and attention is now applied between
        # the two residual blocks (not after them as in the encoder and decoder
        # blocks)
        x1 = self.res_block1(x, time_embs)

        x2 = self.group_norm(x1)
        x2 = self.attention(x2)

        x3 = x1 + x2

        x4 = self.res_block2(x3, time_embs)

        # No residual is returned since this is the bottleneck (does not pass
        # residual connection to decoder)
        return x4

class decoder_block(nn.Module):
    def __init__(self, dim_in, dim_out, time_size, is_last):
        super().__init__()

        self.res_block1 = ResNetBlock(dim_out * 2, dim_in, time_size)
        self.res_block2 = ResNetBlock(dim_in, dim_in, time_size)
        self.group_norm = nn.GroupNorm(1, dim_in)  # TODO: Why only a single group - not e.g. 8?
        self.lin_attention = LinearAttention(dim_in)

        if not is_last:
            self.final = nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1)
        else:
            self.final = nn.Identity()

    def forward(self, x, time_embs):
        # Notice that the residual from the encoder block has already been
        # concatenated to the input x in the UNet forward method (so x is
        # the concatenated original x and residual here)
        x1 = self.res_block1(x, time_embs)
        x1 = self.res_block2(x1, time_embs)

        x2 = self.group_norm(x1)
        x2 = self.lin_attention(x2)

        x3 = x1 + x2

        x4 = self.final(x3)

        return x4

class Unet(nn.Module):
    def __init__(self, im_size, in_channels, dim_mults):
        super().__init__()

        # Hyperparameters
        self.in_channels = in_channels

        """ Part 1: Initialize stem """
        # Initialize stem (with output channel number determined by image size)
        stem_channels = im_size // 3 * 2  # 18 channels
        self.init_conv = nn.Conv2d(in_channels, stem_channels, kernel_size=(3,3), padding=1) # debug: (7,7) with padding 3

        """ Part 2: Initialize time embedding network """
        # Initialize time embedding network with calculation of position embeddings
        # (notice the dimensions of the time embedding network is not the original
        # image dimension - but instead time_size [4 times as large])
        time_size = im_size * 4

        self.time_mlp = nn.Sequential(
            PositionEmbeddings(im_size),     # Output shape: (batch_size,im_size)
            nn.Linear(im_size, time_size),
            nn.GELU(),
            nn.Linear(time_size, time_size), # Output shape: (batch_size,time_size)
        )

        """ Part 3: Determine blocks input/output channel numbers """
        # Define the number of input and output channels for each encoder block in UNet
        dims = [stem_channels, *map(lambda m: im_size * m, dim_mults)] # [18, 28, 56, 112]
        in_out = list(zip(dims[:-1], dims[1:])) # [(18, 28), (28, 56), (56, 112)]

        """ Part 4: Initialize encoder blocks """
        # Module list of encoder blocks corresponding to levels in UNet encoder
        self.downs = nn.ModuleList([])

        for block_idx, (dim_in, dim_out) in enumerate(in_out):
            # Determine if current encoder block is the last one before the bottleneck
            is_last = block_idx >= (len(in_out) - 1)

            # Add encoder block to module list
            self.downs.append(encoder_block(dim_in, dim_out, time_size, is_last))

        """ Part 5: Initialize bottleneck block """
        # Initialize bottleneck block
        self.bottleneck = bottleneck_block(dims[-1], time_size)

        """ Part 6: Initialize decoder blocks """
        # Module list of encoder blocks corresponding to levels in Unet decoder
        self.ups = nn.ModuleList([])

        for block_idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # TODO: Why [1:]?
            # Determine if current decoder block is the last one before the final (non-decoder) block
            is_last = block_idx >= (len(in_out) - 1)

            # Add decoder block to module list
            self.ups.append(decoder_block(dim_in, dim_out, time_size, is_last))

        """ Part 7: Initialize final block """
        # Initialize final block (ResNet block with 1x1 conv) # TODO: Why im_size for channels (surely wrong!)?
        self.final_block = nn.Sequential(
            ResNetBlock(im_size,im_size),
            nn.Conv2d(im_size,in_channels,1),
        )

    def forward(self, x, time_idxs):
        # Send image batch through stem convolution
        x = self.init_conv(x)

        # Get time embeddings for current batch and send them through
        # a small MLP (general encoder of time embeddings)
        time_embs = self.time_mlp(time_idxs)

        # Downsample part of UNet
        residuals = []

        for block in self.downs:
            x, residual = block(x, time_embs)
            residuals.append(residual)

        # Bottleneck part of UNet
        x = self.bottleneck(x,time_embs)

        # Upsample part of UNet
        for block in self.ups:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = block(x, time_embs)

        # Use final block of convolutions + extra convolution to
        # map signal to 1 channel (predicting pixel noise in input
        # image) instead of 3 channels (like input image)
        x = self.final_block(x)

        return x

class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        # Define hyperparameters
        self.im_dim = config.im_dim
        self.channels = config.channels
        self.dim_mults = config.dim_mults
        self.device = config.device
        self.T = config.T

        # Initialize U-Net (for reverse diffusion process)
        self.Unet = Unet(self.im_dim, self.channels, self.dim_mults).to(self.device)

        # Initialize beta schedule (tensor of T beta values)
        self.betas = self.beta_schedule(config.T)

        # Pre-calculate variables for the forward diffusion process
        self.pre_vars = self.pre_calc_vars(self.betas)

    def beta_schedule(self, T):
        """ Function: As used in the second DDPM version I have notes for
            Input:    Number of time steps in forward and reverse diffusion process
            Output:   Tensor of shape (T) with all the values of beta_t
        """

        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return torch.clip(betas, 0.0001, 0.9999)

    def pre_calc_vars(self, betas):
        # Pre-calculate alphas (alpha_t = 1 - beta_t)
        alphas = 1. - betas

        # Pre-calculate bar-alphas (cumulative product of all alphas from time step 0 to t)
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        # Remove last bar-alpha value (at time step T) and instead add 1.0 as first value
        # (at time step 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculate 1/sqrt(alpha_t) values for all time steps
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        ### Calculations for diffusion q(x_t | x_{t-1}) and others
        # These calculations can be recognized from the formulae in the RPs
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        ### Calculations for posterior q(x_{t-1} | x_t, x_0)
        # These calculations can be recognized from the formulae in the RPs
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Collect in a dictionary
        precalc_vars = {'sqrt_recip_alphas': sqrt_recip_alphas,
                        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
                        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
                        'posterior_variance': posterior_variance}

        return precalc_vars

    def extract(self, a, t, x_shape):
        """ Function: Extract appropriate t index for a batch of indices """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward(self, x, time_idxs):
        """ Function: Calculates the loss of the DDPM model (predicted noise vs. actual noise)
            Input: Batch of images x and list of timestep indices (1 for each image in batch)
        """

        """ Part 1: Forward diffusion process (using the nice property) """
        # Fetch sqrt(\bar{\alpha}_t) and \sqrt(1-\bar{\alpha}_t) for all time_idxs (used in nice property)
        sqrt_alphas_cumprod_t = self.extract(self.pre_vars['sqrt_alphas_cumprod'], time_idxs, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.pre_vars['sqrt_one_minus_alphas_cumprod'], time_idxs, x.shape)

        # Compute the noisy images for the different time_idxs using the nice property
        noise = torch.randn_like(x)  # sample from standard Gaussian distribution
        x_noisy = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise # nice property (reparameterization)

        """ Part 2: Reverse diffusion process (using U-Net) """
        predicted_noise = self.Unet(x_noisy, time_idxs)

        """ Part 3: Calculate loss """
        # Notice that the fact that we predict the noise and not the image is fundamentally
        # different from the reverse diffusion process in the textbook DDPM (this version of
        # the MSE loss is not normalizd with the number of observations in the batch)
        loss = F.mse_loss(noise, predicted_noise, reduction='mean')

        return loss

    @torch.no_grad()
    def sample_reverse(self, batch_size):
        """ Function: Generates list of np-array images (number given by batch size) """

        # Start from pure noise (batch of noise images)
        shape = (batch_size, self.channels, self.im_dim, self.im_dim)
        x = torch.randn(shape, device=self.device)

        # Sample time indices from uniform distribution
        for i in tqdm(reversed(range(0, self.T)), total=self.T):
            # Tensor of shape (batch_size) with all time indices equal to i (current time step index)
            time_idxs = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # Get beta value for current time step
            betas_t = self.extract(self.betas, time_idxs, x.shape)

            # Get precalculated variables for forward diffusion process for current time step
            sqrt_one_minus_alphas_cumprod_t = self.extract(self.pre_vars['sqrt_one_minus_alphas_cumprod'], time_idxs, x.shape)
            sqrt_recip_alphas_t = self.extract(self.pre_vars['sqrt_recip_alphas'], time_idxs, x.shape)

            # Use reverse diffusion process to predict mean of data distribution (equation 11 in RP)
            model_mean = sqrt_recip_alphas_t * (x - betas_t * self.Unet(x, time_idxs) / sqrt_one_minus_alphas_cumprod_t)

            if i != 0:
                # If we are still at a intermediate time step in the reverse proces then add the noise from above to
                # noise image from the last time step to get the noise image from this time step (used in next time step)
                posterior_variance_t = self.extract(self.pre_vars['posterior_variance'], time_idxs, x.shape)
                noise = torch.randn_like(x)

                x = model_mean + torch.sqrt(posterior_variance_t) * noise

        return x

def train_epoch(model, optimizer, data_loader, epoch, config):
    model.train()

    # Hold aggregated loss
    total_loss = 0

    with tqdm(data_loader, unit="batch", desc=f" {epoch+1}/{config.num_epochs}") as batch:
        for i, (x, _) in enumerate(batch):
            optimizer.zero_grad()

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            actual_batch_size = x.shape[0]
            time_idxs = torch.randint(0, config.T, (actual_batch_size,), device=config.device).long()

            # Calculate total batch loss (not normalized with batch size)
            batch_loss = model.forward(x.to(config.device), time_idxs)

            # Optimize parameters
            batch_loss.backward()
            optimizer.step()

            # Update metric
            total_loss += batch_loss.item()

    # Return average of batch losses (which is an average in itself)
    epoch_loss = total_loss / (i+1)

    return epoch_loss

def eval_epoch(model, data_loader, config):
    model.eval()

    # Hold aggregated loss
    total_loss = 0

    for i, (x, _) in enumerate(data_loader):
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        actual_batch_size = x.shape[0]
        time_idxs = torch.randint(0, config.T, (actual_batch_size,), device=config.device).long()

        # Calculate loss (not normalized with batch size)
        batch_loss = model.forward(x.to(config.device), time_idxs)

        # Update metric
        total_loss += batch_loss.item()

    # Return average of batch losses (which is an average in itself)
    epoch_loss = total_loss / (i+1)

    return epoch_loss

def main(config):
    # wandb logging
    wandb.init(config=config, project="special_course", entity="louisdt")

    # Load data
    train_loader, val_loader, test_loader = load_data(config)

    # Initialize model
    model = DDPM(config).to(config.device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config.lr)

    # Train model (and test on validation set)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Train and validate 1 epoch
        train_loss = train_epoch(model, optimizer, train_loader, epoch, config)
        val_loss = eval_epoch(model, val_loader, config)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        # Log metrics
        if epoch % 10 == 0:
            # Create 5 samples of generated images
            samples = model.sample_reverse(5)

            # Log metrics
            wandb.log({"epoch": epoch,
                       "train_loss": train_loss,
                       "val_loss": val_loss,
                       "samples": [wandb.Image(samples[i,0,:,:]) for i in range(5)]})
        else:
            wandb.log({"epoch": epoch,
                       "train_loss": train_loss,
                       "val_loss": val_loss})

    # Test best version of model
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss = eval_epoch(model, test_loader, config)

    # Log metrics
    wandb.log({"test_loss": test_loss})

    # Save 10 samples from best model locally
    samples = model.sample_reverse(20)
    for i in range(5):
        plt.imsave(f"advanced_sample_{i}.png", samples[i,0,:,:].cpu(), cmap="gray")

class hyperparameters():
    def __init__(self):
        self.im_dim = 28  # Size of image (28 in MNIST)
        self.channels = 1 # Number of input image channels (1 in MNIST)
        self.dim_mults = (1, 2, 4,) # Multipliers for the number of channels in each layer
        self.T = 50   # Number of diffusion steps
        self.batch_size = 128
        self.lr = 1e-3
        self.num_epochs = 50

        if torch.cuda.is_available():
            self.device = torch.device("cuda")

if __name__ == "__main__":
    config = hyperparameters()
    main(config)

























