# model 2.0 (EEGNet, multi-scale conv + factorised attention).py
import torch
import torch.nn as nn
import math

class MultiScaleConvBlock(nn.Module):
    """multi-scale convolution block, with size of 16, 32, 64, 128"""
    def __init__(self, num_electrodes, F1=8, D=2, kernel_sizes=[16,32,64,128], pool_size=2, downsamples=4):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.D = D
        self.kernel_sizes = kernel_sizes
        self.pool_size = pool_size
        self.downsamples = downsamples

        # multiscale conv
        self.time_convs = nn.ModuleList([
            nn.Conv2d(1, F1, kernel_size=(1, ks), padding=(0, ks//2), bias=False)
            for ks in kernel_sizes
        ])

        # batch norm + elu
        total_f1 = F1 * len(kernel_sizes)
        self.bn1 = nn.BatchNorm2d(total_f1)
        self.act1 = nn.ELU(inplace=True)

        # depthwise
        self.depthwise = nn.Conv2d(total_f1, total_f1 * D,
                                   kernel_size=(num_electrodes, 1),
                                   groups=total_f1, bias=False)
        self.bn2 = nn.BatchNorm2d(total_f1 * D)
        self.act2 = nn.ELU(inplace=True)

        # time points down sampling
        pools = []
        for _ in range(downsamples):
            pools.append(nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)))
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        branch_outs = []
        for conv in self.time_convs:
            out = conv(x)
            branch_outs.append(out)
        x = torch.cat(branch_outs, dim=1)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act2(x)

        for pool in self.pools:
            x = pool(x)
        x = x.squeeze(2)
        return x


class FactorizedTransformer(nn.Module):
    """factorised attention：attention on time, channel separately and then merge"""
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # time block
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)

        # channel block
        channel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.channel_encoder = nn.TransformerEncoder(channel_layer, num_layers=num_layers)

        # positional encoding（sin encoding + channel encoding(learn)）
        self.register_buffer('temporal_pos', self._get_sinusoidal_pos_enc(512, hidden_dim))
        self.channel_pos = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def _get_sinusoidal_pos_enc(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x_time, x_chan):
        """
        x_time: (B, T, hidden_dim)  time block input
        x_chan: (B, H, hidden_dim)  channel block input
        """
        # time attention
        T = x_time.size(1)
        pos = self.temporal_pos[:, :T, :].to(x_time.device)
        x_time = x_time + pos
        x_time = self.temporal_encoder(x_time)   # (B, T, hidden_dim)

        # channel attention
        x_chan = x_chan + self.channel_pos.to(x_chan.device)
        x_chan = self.channel_encoder(x_chan)    # (B, H, hidden_dim)

        # merge both：transpose time block output and fuse with channel block output
        x_fused = x_time.permute(0, 2, 1) + x_chan   # (B, hidden_dim, H)
        x_fused = x_fused.permute(0, 2, 1)           # (B, H, hidden_dim)
        return x_fused


class EEGNetEnhanced(nn.Module):
    """enhanced EEGNet has both multi-scalr conv + factorised attention"""
    def __init__(self, chunk_size, num_electrodes, F1=8, F2=16, D=2, num_classes=2,
                 kernel_1=64, kernel_2=16, dropout=0.3):
        super().__init__()
        self.conv = MultiScaleConvBlock(
            num_electrodes=num_electrodes,
            F1=F1,
            D=D,
            kernel_sizes=[16,32,64,128],
            pool_size=2,
            downsamples=4
        )

        # calculate the channel after squeeze
        self.H = F1 * len([16,32,64,128]) * D   # 64
        self.T_prime = chunk_size // (2 ** 4)    # 128

        # project layer: projects both to hidden_dim
        hidden_dim = 128
        self.time_proj = nn.Linear(self.H, hidden_dim)
        self.chan_proj = nn.Linear(self.T_prime, hidden_dim)

        # Factorised Transformer
        self.transformer = FactorizedTransformer(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )

        # project from h_hidden to H
        self.out_proj = nn.Linear(hidden_dim, self.H)

        # binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.H, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)

        B, H, Tp = x.shape
        # time block proj: (B, T', H)==>(B, T', hidden_dim)
        x_time = x.permute(0, 2, 1)
        x_time = self.time_proj(x_time)

        # channel block proj: (B, H, T')==>(B, H, hidden_dim)
        x_chan = x.permute(0, 1, 2)
        x_chan = self.chan_proj(x_chan)

        # 3. Factorised Transformer
        x_fused = self.transformer(x_time, x_chan)

        # 4. merge
        x_pool = x_fused.mean(dim=1)

        # 5. project back to H
        x_pool = self.out_proj(x_pool)

        # 6. classification
        out = self.classifier(x_pool)
        return out



def create_eegnet(chunk_size, num_electrodes, F1, F2, D, num_classes,
                  kernel_1, kernel_2, dropout):
    model = EEGNetEnhanced(
        chunk_size=chunk_size,
        num_electrodes=num_electrodes,
        F1=F1,
        F2=F2,
        D=D,
        num_classes=num_classes,
        dropout=dropout
    )
    return model
