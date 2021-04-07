import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model import Generator


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    noise = torch.randn(64, 100, 1, 1, device=device)
    state_dict = torch.load('generator.pth', map_location=device)
    generator = Generator(ngpu=1).to(device)
    generator.load_state_dict(state_dict)
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    fake_images = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
    plt.figure()
    plt.axis('off')
    plt.title('Fake Images')
    plt.imshow(np.transpose(fake_images, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test()
