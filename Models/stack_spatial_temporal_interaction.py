import math
import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args):
        if len(args) == 0:
            return self.fn(self.norm(x))
        else:
            return self.fn(self.norm(x), self.norm(args[0]))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialTransformer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.spatial_attention = Attention(dim, heads, dim_head, dropout, spatial=True)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, phys_x):
        spatial_attention, v = self.spatial_attention(x, phys_x)
        spatial_attention = self.dropout(spatial_attention)
        out = torch.matmul(spatial_attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.temporal_attention = Attention(dim, heads, dim_head, dropout, spatial=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, phys_x):
        temporal_attention, v = self.temporal_attention(x, phys_x)
        temporal_attention = self.dropout(temporal_attention)
        out = torch.matmul(temporal_attention, v)
        out = rearrange(out, 'b h t d -> b t (h d)')

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=4, dropout=0., spatial=False):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(dim, inner_dim * 1, bias=False)

    def forward(self, x, phys_x):
        qk = self.to_qk(phys_x).chunk(2, dim=-1)
        v = self.to_v(x).chunk(1, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)
        v = rearrange(v[0], 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        return attn, v
    

class FreModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed_size = dim
        self.scale = 0.02
        self.sparsity_threshold = 0.01
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)
        # sampling rate of facial video
        self.sampling_rate = 30
        # Low frequency of physiological signal bandwidth
        self.low_freq = 0.66
        # High frequency of physiological signal bandwidth
        self.high_freq = 3.0
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')
    
    def topk_freq(self, x_freq):
        top_k = 8
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask
        
    def forward(self, x, layer):
        # input x shape is [bs, T, dm]
        bs, T, dm = x.shape
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        

        if layer != 0:
            # PBF
            rfreqs = torch.fft.rfftfreq(T, 1/self.sampling_rate)
            pass_f = (torch.abs(rfreqs) >= self.low_freq) & (torch.abs(rfreqs) <= self.high_freq)
            pass_f = pass_f.unsqueeze(0).unsqueeze(-1).to(x.device)
            x = x * pass_f

        # PSM
        origin_ffted = x
        o1_real = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device)
        o1_imag = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device)
        o1_real = F.relu(
            torch.einsum('bld,de->ble', x.real, self.r1) + torch.einsum('bld,de->ble', x.imag, self.i1) + self.rb1
        )
        o1_imag = F.relu(
            torch.einsum('bld,de->ble', x.real, self.i1) - torch.einsum('bld,de->ble', x.imag, self.r1) + self.ib1
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, self.sparsity_threshold)
        y = torch.view_as_complex(y)
        y = y * origin_ffted

        # ASS
        freq_mask = self.create_adaptive_high_freq_mask(y)
        y = y * freq_mask.to(x.device)

        y = torch.fft.irfft(y.transpose(-1, -2), norm='ortho').abs().transpose(-1, -2)
        y = y[:, :T, :]
        return y


class SpatialTemporalInteraction(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        # self.fre_modulation = fre_modulation(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                FreModulation(dim),
                PreNorm(dim, SpatialTransformer(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                FreModulation(dim),
                PreNorm(dim, TemporalTransformer(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.pos_embedding = nn.Parameter(torch.randn(1, 300, dim))

    def forward(self, x, phys_x):
        layer = 0
        for fre_modulation_s, spatial_attn, ff1, fre_modulation_t, temporal_att, ff2 in self.layers:
            if layer == 0:
                phys_x = rearrange(phys_x, '(B T) N D -> (B N) T D', T=300)
            else:
                phys_x = rearrange(x, '(B T) N D -> (B N) T D', T=300)

            phys_x = fre_modulation_s(phys_x, layer)

            phys_x = rearrange(phys_x, '(B N) T D -> (B T) N D', N=63)
            x = spatial_attn(x, phys_x) + x
            x = ff1(x) + x
            
            x = rearrange(x, '(B T) N D -> (B N) T D', T=300)

            phys_x = rearrange(phys_x, '(B T) N D -> (B N) T D', T=300)
            if layer == 0:
                x += self.pos_embedding[:, :300]
                x = self.dropout(x)
            x = temporal_att(x, phys_x) + x
            x = ff2(x) + x
                
            x = rearrange(x, '(B N) T D -> (B T) N D', N=63)
            layer = layer + 1

        return x



if __name__ == "__main__":
    pass
