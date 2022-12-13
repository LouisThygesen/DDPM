import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.distributions import Normal

import numpy as np
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

class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()

        # Hyperparameters
        self.device = config.device                                     # Device to use (default: GPU)
        self.in_dim = config.in_dim                                     # Dim of input (MNIST is 28x28 = 784)
        self.hid_dim = config.hid_dim                                   # Dim of hidden layers
        self.T = config.T                                               # Number of diffusion steps
        self.beta = torch.FloatTensor([config.beta]).to(self.device)    # Beta from research paper (not as in VAE loss)

        # List of networks that each parameterize a distribution p(z_i|z_i+1) [except p(x|z_1)]
        self.p_dnns = nn.ModuleList([nn.Sequential(nn.Linear(self.in_dim, self.hid_dim), nn.LeakyReLU(),
                                                   nn.Linear(self.hid_dim, self.hid_dim), nn.LeakyReLU(),
                                                   nn.Linear(self.hid_dim, self.hid_dim), nn.LeakyReLU(),
                                                   nn.Linear(self.hid_dim, self.in_dim*2)) for _ in range(self.T - 1)])

        # Add newtork that parameterizes the distribution p(x|z_1)
        self.p_dnns.append(nn.Sequential(nn.Linear(self.in_dim, self.hid_dim*2), nn.LeakyReLU(),
                                         nn.Linear(self.hid_dim*2, self.hid_dim*2), nn.LeakyReLU(),
                                         nn.Linear(self.hid_dim*2, self.hid_dim*2), nn.LeakyReLU(),
                                         nn.Linear(self.hid_dim*2, self.in_dim), nn.Tanh()))

    def forward(self, x):
        """ Part 1: Forward difussion process """
        # The forward process conditionals q(x_t|x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I) is used to sample x_t
        # from x_{t-1} iteratively (starting with x_0 as input).
        dist = Normal(torch.sqrt(1. - self.beta) * x, torch.sqrt(self.beta))
        zs = [dist.sample()]

        for step in range(1, self.T):
            dist = Normal(torch.sqrt(1. - self.beta) * zs[-1], torch.sqrt(self.beta))
            zs.append(dist.sample())

        """ Part 2: Backward diffusion process """
        # The reverse process conditionals p(x_{t-1}|x_t) = N(x_t; mu(x_{t-1}), exp(log(var(x_t)))*I) are predicted
        # using the latents x_t from the forward process (no sampling) for the transtions between steps T to 1 (second
        # last transition)
        mus = []
        log_vars = []

        for i, step in enumerate(reversed(range(1, self.T))):
            h = self.p_dnns[i](zs[step])
            mu, log_var = torch.chunk(h, 2, dim=1)
            mus.append(mu)
            log_vars.append(log_var)

        # Predicting the last transition: outputting the means for x as we assume the last distribution is
        # Normal(x_0|tanh(NN(x_1)), I)
        mu_x = self.p_dnns[-1](zs[0])

        """ Part 3: Compute loss """
        # We calculate ln(p(x_{0,i}|x_1)) for every pixel i in x_0 and sum over all pixels [as log(a)+log(b)=log(a*b)]
        # to get ln(p(x_0|x_1)) [log probability of entire image - Monte Carlo sample estimate of reconstruction loss]
        dist = Normal(mu_x, torch.ones_like(mu_x).to(self.device))
        RE = dist.log_prob(x).sum(-1)

        # Variable to collect KL divergence estimates (Monte Carlo estimate of expectation in the definition of
        # KL divergence)
        KL = 0

        # Summation symbol part of ELBO (plus extra loss part from book which
        # is written outside of summation symbol only ue to the bad notation)
        for i in range(len(mus)):
            # p(x_{t-1}|x_t) = N(x_{t-1}; mu(x_{t-1}), exp(log(var(x_t)))*I)
            normal1 = Normal(mus[i], torch.exp(0.5 * log_vars[i]))

            # q(x_t|x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)
            normal2 = Normal(torch.sqrt(1. - self.beta) * zs[i], torch.sqrt(self.beta))

            # Monte Carlo estimate expectation in KL divergence definition
            KL += normal2.log_prob(zs[i]).sum(-1) - normal1.log_prob(zs[i]).sum(-1)

        # KL divergence between baseline p(z_t) and q(z_T|z_{T-1})
        normal1 = Normal(torch.zeros_like(zs[-1]).to(self.device), torch.ones_like(zs[-1]).to(self.device))
        normal2 = Normal(torch.sqrt(1. - self.beta) * zs[-1], torch.sqrt(self.beta))
        KL += normal2.log_prob(zs[-1]).sum(-1) - normal1.log_prob(zs[-1]).sum(-1)

        # Final ELBO
        loss = - (RE - KL).mean()

        return loss, -RE.mean()

    def sample_reverse(self, batch_size):
        # Sample from base distribution (multivariate standard Gaussian)
        z = torch.randn([batch_size, self.in_dim]).to(self.device)

        # Run reverse diffusion process (all except last transition) with sampling at each step (unlike in training)
        for i, step in enumerate(reversed(range(1, self.T))):
            h = self.p_dnns[i](z)
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            dist = Normal(mu_i, torch.exp(0.5 * log_var_i))
            z = dist.sample()

        # Predict ean for the last distribution Normal(x|mu_x, 1) (last transition conditional distribution). Instead
        # of sampling from the distribution we just use the mean as a sample as it is the most likely sample.
        mu_x = self.p_dnns[-1](z)

        # Transform from [-1,1] back to 256 gray scale
        x = (mu_x + 1) / 2
        x = x * 255

        # Reshape to 28x28
        x = x.view(batch_size, 28, 28)

        return x

    def sample_forward(self, x):
        # Run full forward diffusion process to get noise sample x_T
        dist = Normal(torch.sqrt(1. - self.beta) * x, torch.sqrt(self.beta))
        zs = [dist.rsample()]

        for step in range(1, self.T):
            dist = Normal(torch.sqrt(1. - self.beta) * zs[-1], torch.sqrt(self.beta))
            zs.append(dist.sample())

        return zs[-1]

def load_data(config):
    """ Function: Load dataset partitions and apply transformations
        Input:    Batch size
        Output:   Train, val and test loader
    """

    # Data transformations
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: torch.flatten(x)),    # flatten image
        T.Lambda(lambda x: 2. * x - 1.)  # rescale to [-1, 1]
    ])

    # Initialize dataset
    train_data = MNIST(root='data', train=True, download=True, transform=transforms)
    train_data, val_data = random_split(train_data, [55000,5000], generator=torch.Generator().manual_seed(42))
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

def train_epoch(model, optimizer, data_loader, epoch, config):
    model.train()

    # Hold aggregated loss
    total_loss = 0
    total_RE = 0

    with tqdm(data_loader, unit="batch", desc=f" {epoch+1}/{config.num_epochs}") as batch:
        for i, (x, _) in enumerate(batch):
            # Calculate loss, gradient and take optimization step
            optimizer.zero_grad()
            batch_loss, RE = model.forward(x.to(config.device))
            batch_loss.backward()
            optimizer.step()

            # Update metric
            total_loss += batch_loss.item()
            total_RE += RE.item()

    # Return average of batch losses (which is an average in itself)
    epoch_loss = total_loss / (i+1)
    epoch_RE = total_RE / (i+1)

    return epoch_loss, epoch_RE

def eval_epoch(model, data_loader, config):
    model.eval()

    # Hold aggregated loss
    total_loss = 0
    total_RE = 0

    for i, (x, _) in enumerate(data_loader):
        # Calculate loss
        batch_loss, RE = model.forward(x.to(config.device))

        # Update metric
        total_loss += batch_loss.item()
        total_RE += RE.item()

    # Calculate average loss per observation
    epoch_loss = total_loss / (i+1)
    epoch_RE = total_RE / (i+1)

    return epoch_loss, epoch_RE

def main(config):
    # wandb logging
    wandb.init(config=config, project="special_course", entity="louisdt")

    # Load data
    train_loader, val_loader, test_loader = load_data(config)

    # Initialize model
    model = DDPM(config).to(config.device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train model (and test on validation set)
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Train and validate 1 epoch
        train_loss, train_RE = train_epoch(model, optimizer, train_loader, epoch, config)
        val_loss, val_RE = eval_epoch(model, val_loader, config)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f'New best loss: {val_loss}')

        # Log metrics
        if epoch % 3 == 0:
            # Create 5 samples of generated images
            samples = model.sample_reverse(5)

            # Log metrics
            wandb.log({"epoch": epoch,
                       "train_loss": train_loss,
                       "val_loss": val_loss,
                       "train_RE": train_RE,
                       "val_RE": val_RE,
                       "samples": [wandb.Image(sample.reshape(28, 28)) for sample in samples]})
        else:
            wandb.log({"epoch": epoch,
                       "train_loss": train_loss,
                       "val_loss": val_loss,
                       "train_RE": train_RE,
                       "val_RE": val_RE})

    # Test best version of model
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_RE = eval_epoch(model, test_loader, config)

    # Log metrics
    wandb.log({"test_loss": test_loss,
               "test_RE": test_RE})

    # Save 10 samples from model locally
    samples = model.sample_reverse(20)
    for i, sample in enumerate(samples):
        plt.imsave(f"textbook_sample_{i}.png", sample.detach().cpu(), cmap="gray")

class hyperparameters():
    in_dim = 784        # Input dimension (MNIST is 28x28=784)
    T = 5               # Number of diffusion steps
    beta = 0.9          # Beta from research paper (not as in VAE loss)
    hid_dim = 256       # Used to control the size of the hidden dimension
    lr = 1e-3
    num_epochs = 50
    batch_size = 256

    if torch.cuda.is_available():
        device = torch.device("cuda")

if __name__ == "__main__":
    config = hyperparameters()
    main(config)



















