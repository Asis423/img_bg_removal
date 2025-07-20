import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    """Downsampling block: ConvBlock -> MaxPool"""
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip

class UpSample(nn.Module):
    """Upsampling block: ConvTranspose -> Concatenate -> ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """Custom U-Net implementation from scratch"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DownSample(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        
        # Decoder (Expanding Path)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(UpSample(feature*2, feature))
        
        # Final classifier
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, up in enumerate(self.ups):
            x = up(x, skip_connections[idx])
        
        # Final output
        return torch.sigmoid(self.final_conv(x))

# Test the model
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 320, 320)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")