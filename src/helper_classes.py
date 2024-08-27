import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    """Loss function for regression tasks"""
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = (y_t - y_prime_t)/3 # divide by 3 to avoid numerical overflow in cosh
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
    
    
class Dataset:
    def __init__(self, data_x, data_y=None):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        try:
            if self.data_y is not None:
                return self.data_x[idx], self.data_y[idx]
            else:
                return self.data_x[idx]
        except Exception as e:
            print(f"Error accessing data at index {idx}: {str(e)}")
            raise
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return torch.sigmoid(out).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x