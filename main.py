import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
from torch.nn.modules.activation import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
import os

class Discriminator(nn.Module):

  # 784 in features
  def __init__(self, img_dim):
    super().__init__()

    self.disc = nn.Sequential(
        nn.Linear(img_dim, 128),

        # slope of 0.1. LeakyRelu is preferred for GAN's
        nn.LeakyReLU(0.1),

        # 1 out is fake = 0, real = 1
        nn.Linear(128, 1),

        # ensure the value is between 0 and 1
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.disc(x)


class Generator(nn.Module):

  # z_dim is the dim of the noise the generator takes in as input
  def __init__(self, z_dim, img_dim):
    super().__init__()
    self.gen = nn.Sequential(
        nn.Linear(z_dim, 256),
        nn.LeakyReLU(0.1),
        nn.Linear(256, img_dim),

        # make sure output of pixel values are between -1,1
        # we will normalize the data from mnist to be between -1,1
        nn.Tanh()
    )

  def forward(self, x):
    return self.gen(x)

def main():
    # hyperparams
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64 # try other idms as well
    image_dim = 28 * 28 * 1 # 784
    batch_size = 32
    num_epochs = 50

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)

    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    transforms = transforms.Compose(
        # mean and stdv of the mnist dataset
        [transforms.ToTensor(), transforms.Normalize((0.5), 0.5)]
    )

    dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)

    criterion = nn.BCELoss()

    # outputs fake images from the generator
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0

    def show(img):
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()

    if int(os.environ["COLAB_GPU"]) > 0:
        print("a GPU is connected.")
    elif "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
        print("A TPU is connected.")
    else:
        print("No accelerator is connected.")

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            # keep the number of rows in the batch but flatten it so its
            # 784 cols

            # this is the target image
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # *** train the discrimintor

            # create some noisy image
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)

            # feed the disc model the real image
            disc_real = disc(real).view(-1)

            # the real image should be compared against a matrix of 1's because
            # thats where the pixels should be lighting up
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            # the disc should be compared to a fake image against all zeros
            # because it's not real
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # train the generator
            output = disc(fake).view(-1)

            # now the disc has been trained a bit mroe on producing images
            # closer to what is real
            # giving the disc model a fake image should make it closer to the
            # real thing
            # so anywhere the output isn't 1 should be considered a loss
            # and then use this output to train the gen
            # you're using the disc as the output because you want to train the
            # gen in the dir of the disc model. You want to make it
            # be able to produce images that fool the disc
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                            Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                print("Generated image:")
                print(img_grid_fake.shape)
                show(img_grid_fake)
                show(img_grid_real)
                # plt.imshow(img_grid_fake.permute(1, 2, 0))

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
main()