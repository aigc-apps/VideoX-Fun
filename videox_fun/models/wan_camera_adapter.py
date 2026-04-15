import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block with two convolutions and skip connection."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class SimpleAdapter(nn.Module):
    """Adapter module that processes camera information through spatial downsampling and feature extraction."""
    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor=8, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()
        
        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        
        # Convolution: reduce spatial dimensions by a factor of 2
        self.conv = nn.Conv2d(in_dim * downscale_factor * downscale_factor, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        
        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)
        
        Returns:
            Processed tensor of shape (batch, out_dim, frames, new_height, new_width)
        """
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        
        # Apply pixel unshuffle and convolution
        x_unshuffled = self.pixel_unshuffle(x)
        x_conv = self.conv(x_unshuffled)
        
        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)
        
        # Reshape to restore original batch-frame structure
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))
        
        # Permute to (batch, channels, frames, height, width)
        out = out.permute(0, 2, 1, 3, 4)

        return out