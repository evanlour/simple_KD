import torch

class SimpleGenerator(torch.nn.Module):
    def __init__(self, img_size: int, latent_size: int, channels: int, channel_format=torch.channels_last):
        """A simple image generator used for methods such as DAFL, DFAD, ZSKT etc.

        Args:
            img_size (int): The target image size.
            latent_size (int): The latent dimension size.
            channels (int): The target image channels.
        """
        super(SimpleGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = torch.nn.Linear(latent_size, 128 * self.init_size**2)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.channel_format = channel_format

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, momentum=0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, momentum=0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.reshape(-1, 128, self.init_size, self.init_size)
        out = out.to(memory_format=self.channel_format)
        out = self.bn1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2)
        out = self.block1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2)
        out = self.block2(out)
        return out
