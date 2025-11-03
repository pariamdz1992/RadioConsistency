"""
Exact conditioning U-Net adapted from diffusion model for consistency training.
This preserves the exact conditioning architecture: Swin backbone + RelationNet cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, reduce

# Import the real backbone networks from copied files
from cm.swin_transformer import swin_b, Swin_B_Weights
from cm.efficientnet import efficientnet_b7, EfficientNet_B7_Weights
from cm.resnet import resnet101, ResNet101_Weights
from cm.vgg import vgg16, VGG16_Weights


class PositionEmbeddingSine(nn.Module):
    """Position embedding for attention - exact copy from diffusion model."""
    
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        num_feats = x.shape[3]
        num_pos_feats = num_feats // 2
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device).to(torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).contiguous()
        return pos


class Mlp(nn.Module):
    """MLP - exact copy from diffusion model."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BasicAttetnionLayer(nn.Module):
    """Cross-attention layer - exact copy from diffusion model."""
    
    def __init__(self, embed_dim=128, nhead=8, ffn_dim=512, window_size1=[4, 4],
                 window_size2=[1, 1], dropout=0.1):
        super().__init__()
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.avgpool_q = nn.AvgPool2d(kernel_size=window_size1)
        self.avgpool_k = nn.AvgPool2d(kernel_size=window_size2)
        self.softmax = nn.Softmax(dim=-1)
        self.nhead = nhead

        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=ffn_dim, drop=dropout)
        self.pos_enc = PositionEmbeddingSine(embed_dim)
        self.concat_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1)
        self.gn = nn.GroupNorm(8, embed_dim)
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2): # x1 for q (conditional input), x2 for k,v
        B, C1, H1, W1 = x1.shape
        _, C2, H2, W2 = x2.shape
        
        shortcut = x2 + self.concat_conv(torch.cat(
            [F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=True),
             x2], dim=1))
        shortcut = self.gn(shortcut)
        
        pad_l = pad_t = 0
        pad_r = (self.window_size1[1] - W1 % self.window_size1[1]) % self.window_size1[1]
        pad_b = (self.window_size1[0] - H1 % self.window_size1[0]) % self.window_size1[0]
        x1 = F.pad(x1, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, H1p, W1p = x1.shape
        
        pad_l = pad_t = 0
        pad_r = (self.window_size2[1] - W2 % self.window_size2[1]) % self.window_size2[1]
        pad_b = (self.window_size2[0] - H2 % self.window_size2[0]) % self.window_size2[0]
        x2 = F.pad(x2, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, H2p, W2p = x2.shape
        
        x1_s = self.avgpool_q(x1)
        qg = self.avgpool_q(x1).permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C2)
        
        kg = self.avgpool_k(x2).permute(0, 2, 3, 1).contiguous()
        kg = kg + self.pos_enc(kg)
        kg = kg.view(B, -1, C1)
        
        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        
        qg = self.q_lin(qg).reshape(B, num_window_q, self.nhead, C1 // self.nhead).permute(0, 2, 1, 3).contiguous()
        kg2 = self.k_lin(kg).reshape(B, num_window_k, self.nhead, C1 // self.nhead).permute(0, 2, 1, 3).contiguous()
        vg = self.v_lin(kg).reshape(B, num_window_k, self.nhead, C1 // self.nhead).permute(0, 2, 1, 3).contiguous()
        kg = kg2
        
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C1)
        qg = qg.transpose(1, 2).reshape(B, C1, H1p // self.window_size1[0], W1p // self.window_size1[1])
        
        x1_s = x1_s + qg
        x1_s = x1_s + self.mlp(x1_s)
        x1_s = F.interpolate(x1_s, size=(H2, W2), mode='bilinear', align_corners=True)
        x1_s = shortcut + self.out_conv(x1_s)
        
        return x1_s


class RelationNet(nn.Module):
    """RelationNet - exact copy from diffusion model."""
    
    def __init__(self, in_channel1=128, in_channel2=128, nhead=8, layers=1, embed_dim=128, ffn_dim=512,
                 window_size1=[4, 4], window_size2=[1, 1]):
        super().__init__()
        self.layers = layers
        self.input_conv1 = nn.Sequential(
            nn.Conv2d(in_channel1, embed_dim, 1),
            nn.BatchNorm2d(embed_dim, momentum=0.03, eps=0.001),
        )
        self.input_conv2 = nn.Sequential(
            nn.Conv2d(in_channel2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim, momentum=0.03, eps=0.001),
        )
        
        self.attentions = nn.ModuleList()
        for i in range(layers):
            self.attentions.append(
                BasicAttetnionLayer(embed_dim=embed_dim, nhead=nhead, ffn_dim=ffn_dim,
                                   window_size1=window_size1, window_size2=window_size2, dropout=0.1)
            )

    def forward(self, cond, feat):
        cond = self.input_conv1(cond)
        feat = self.input_conv2(feat)
        for att in self.attentions:
            feat = att(cond, feat)
        return feat


# Import consistency model building blocks
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class WeightStandardizedConv2d(nn.Conv2d):
    """Weight standardization from diffusion model."""
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ConditionedConsistencyUNet(nn.Module):
    """
    Consistency U-Net with exact conditioning from diffusion model.
    This combines consistency model architecture with diffusion model conditioning.
    """
    
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        # Conditioning parameters (exact from diffusion model)
        cond_in_dim=3,
        cond_dim=128,
        cond_net='swin',  # 'swin', 'effnet', 'resnet', 'vgg'
        fix_bb=True,
        # Standard parameters
        channels=1,
        resnet_block_groups=8,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
        window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],
        fourier_scale=16,
        input_size=[256, 256],
        **kwargs
    ):
        super().__init__()
        
        self.channels = channels
        input_channels = channels
        init_dim = default(init_dim, dim)
        
        # EXACT conditioning backbone from diffusion model
        if cond_net == 'swin':
            f_condnet = 128
            self.init_conv_mask = swin_b(weights=Swin_B_Weights.DEFAULT)
        elif cond_net == 'effnet':
            f_condnet = 48
            self.init_conv_mask = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        elif cond_net == 'resnet':
            f_condnet = 256
            self.init_conv_mask = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif cond_net == 'vgg':
            f_condnet = 128
            self.init_conv_mask = vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            raise NotImplementedError(f"Conditioning network {cond_net} not supported")
        
        # Initial convolution with conditioning
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels + f_condnet, init_dim, 7, padding=3),
            nn.GroupNorm(num_groups=min(init_dim // 4, 8), num_channels=init_dim),
        )
        
        # Dimension setup
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # EXACT feature projection from diffusion model
        self.projects = nn.ModuleList()
        if cond_net == 'effnet':
            self.projects.append(nn.Conv2d(48, dims[0], 1))
            self.projects.append(nn.Conv2d(80, dims[1], 1))
            self.projects.append(nn.Conv2d(224, dims[2], 1))
            self.projects.append(nn.Conv2d(640, dims[3], 1))
        elif cond_net == 'vgg':
            self.projects.append(nn.Conv2d(128, dims[0], 1))
            self.projects.append(nn.Conv2d(256, dims[1], 1))
            self.projects.append(nn.Conv2d(512, dims[2], 1))
            self.projects.append(nn.Conv2d(512, dims[3], 1))
        else:  # swin, resnet
            self.projects.append(nn.Conv2d(f_condnet, dims[0], 1))
            self.projects.append(nn.Conv2d(f_condnet*2, dims[1], 1))
            self.projects.append(nn.Conv2d(f_condnet*4, dims[2], 1))
            self.projects.append(nn.Conv2d(f_condnet*8, dims[3], 1))
        
        # Time embeddings (consistency model style)
        time_dim = dim * 4
        if random_fourier_features:
            sinu_pos_emb = GaussianFourierProjection(dim//2, scale=fourier_scale)
            fourier_dim = dim
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
            
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        # Downsampling path with EXACT conditioning
        self.downs = nn.ModuleList([])
        self.relation_layers_down = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            
            # EXACT RelationNet from diffusion model
            self.relation_layers_down.append(RelationNet(
                in_channel1=dims[ind], 
                in_channel2=dims[ind], 
                nhead=8,
                layers=1, 
                embed_dim=dims[ind], 
                ffn_dim=dims[ind]*2,
                window_size1=window_sizes1[ind], 
                window_size2=window_sizes2[ind]
            ))
        
        # Middle block
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        # Upsampling path with EXACT conditioning
        self.ups = nn.ModuleList([])
        self.relation_layers_up = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            
            dims_rev = dims[::-1]
            self.relation_layers_up.append(RelationNet(
                in_channel1=dims_rev[ind+1], 
                in_channel2=dims_rev[ind],
                nhead=8, 
                layers=1, 
                embed_dim=dims_rev[ind],
                ffn_dim=dims_rev[ind] * 2,
                window_size1=window_sizes1[::-1][ind],
                window_size2=window_sizes2[::-1][ind]
            ))
        
        # Final output
        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
        # Freeze conditioning backbone if specified
        if fix_bb:
            for n, p in self.init_conv_mask.named_parameters():
                p.requires_grad = False

    def forward(self, x, timesteps, cond=None, **kwargs):
        """
        Forward pass with EXACT conditioning flow from diffusion model.
        
        Args:
            x: Input radio maps [B, 1, H, W]
            timesteps: Consistency model timesteps [B]  
            cond: Conditioning [B, 3, H, W] (buildings, transmitters, cars)
        """
        if cond is None:
            raise ValueError("Conditioning is required for conditioned consistency model")
        
        # EXACT conditioning processing from diffusion model
        hm = self.init_conv_mask(cond)  # Multi-scale features from backbone
        
        # Project conditioning features to match U-Net dimensions
        for i, layer in enumerate(self.projects):
            hm[i] = layer(hm[i])
        
        # Initial fusion: concatenate first scale with input (EXACT from diffusion)
        x = self.init_conv(torch.cat([x, F.interpolate(hm[0], size=x.shape[-2:], mode="bilinear")], dim=1))
        r = x.clone()  # Residual connection
        
        # Time embedding
        t = self.time_mlp(timesteps)
        
        # Downsampling with EXACT cross-attention
        h = []
        for i, ((block1, block2, attn, downsample), relation_layer) in enumerate(zip(self.downs, self.relation_layers_down)):
            x = block1(x, t)
            h.append(x)
            
            # EXACT cross-attention with conditioning
            x = relation_layer(hm[i], x)
            
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        # Middle block
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Upsampling with EXACT cross-attention
        for (block1, block2, attn, upsample), relation_layer in zip(self.ups, self.relation_layers_up):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            # EXACT cross-attention with conditioning
            x = relation_layer(hm.pop(), x)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        # Final output
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        return x
