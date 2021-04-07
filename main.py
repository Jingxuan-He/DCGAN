import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from model import Generator, Discriminator
from utils import set_seed, weights_init


# configuration
data_root = 'data'
batch_size = 128
image_size = 64
num_epochs = 5
lr = 0.0002
real_label = 1
fake_label = 0
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def train():
    # create the dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(data_root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create the generator
    g_net = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        g_net = nn.DataParallel(g_net, list(range(ngpu)))
    g_net.apply(weights_init)

    # create the discriminator
    d_net = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        d_net = nn.DataParallel(d_net, list(range(ngpu)))
    d_net.apply(weights_init)

    # create the loss function
    criterion = nn.BCELoss()

    # create the optimizer
    d_optimizer = optim.Adam(d_net.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(g_net.parameters(), lr=lr, betas=(0.5, 0.999))

    # train the model
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real images
            d_net.zero_grad()

            real_images = data[0].to(device)
            real_labels = torch.full((data[0].size(0),), real_label, dtype=torch.float32, device=device)
            output = d_net(real_images).view(-1)
            d_loss_real = criterion(output, real_labels)
            d_loss_real.backward()
            d_x = output.mean().item()

            # train with fake images
            noise = torch.randn(data[0].size(0), 100, 1, 1, device=device)
            fake_images = g_net(noise)
            fake_labels = torch.full((data[0].size(0),), fake_label, dtype=torch.float32, device=device)
            output = d_net(fake_images.detach()).view(-1)
            d_loss_fake = criterion(output, fake_labels)
            d_loss_fake.backward()
            d_g_z1 = output.mean().item()

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g_net.zero_grad()

            output = d_net(fake_images).view(-1)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_optimizer.step()
            d_g_z2 = output.mean().item()

            # output training stats
            if i % 50 == 0:
                print(f'[{epoch + 1}/{num_epochs}][{i + 1}/{len(dataloader)}]\t'
                      f'd_loss: {d_loss:.4f}\t'
                      f'g_loss: {g_loss:.4f}\t'
                      f'd(x): {d_x:.4f}\t'
                      f'd(g(z)): {d_g_z1:.4f}/{d_g_z2:.4f}')

            # save losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            # save the generator
            torch.save(g_net.state_dict(), 'generator.pth')

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(d_losses, label="D")
    plt.plot(g_losses, label="G")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    set_seed(seed=999)
    train()
