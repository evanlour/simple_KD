import torch


class SimpleGenerator(torch.nn.Module):
    """A convolutional generator network used in data-free distillation methods (DAFL, DFAD, ZSKT).

    Maps random Gaussian noise vectors z in R^latent_size to synthetic RGB images in [-1, 1].
    Uses transposed/upsampled convolutions with BatchNorm, LeakyReLU, and Tanh activation.

    Args:
        img_size (int): Target square image resolution (e.g. 32 for CIFAR).
        latent_size (int): Dimension of the input latent noise vector z.
        channels (int): Number of output image color channels (e.g. 3 for RGB).
        channel_format (torch.memory_format, optional): PyTorch memory layout (e.g. channels_last). Defaults to torch.channels_last.
    """

    def __init__(
        self,
        img_size: int,
        latent_size: int,
        channels: int,
        channel_format=torch.channels_last,
    ):
        super().__init__()

        self.init_size = img_size // 4
        self.l1 = torch.nn.Linear(latent_size, 128 * self.init_size**2)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.channel_format = channel_format

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, momentum=0.1),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, momentum=0.1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generates synthetic images from latent vectors.

        Args:
            z (torch.Tensor): Latent noise tensor of shape [batch_size, latent_size].

        Returns:
            torch.Tensor: Generated synthetic images of shape [batch_size, channels, img_size, img_size] with values in [-1, 1].
        """
        out = self.l1(z)
        out = out.reshape(-1, 128, self.init_size, self.init_size)
        out = out.to(memory_format=self.channel_format)
        out = self.bn1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2)
        out = self.block1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2)
        out = self.block2(out)
        return out
