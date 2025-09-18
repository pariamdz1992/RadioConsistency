from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce

# Import original CM components
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

# Import original classes from unet.py
from .unet import (
    AttentionPool2d,
    TimestepBlock,
    TimestepEmbedSequential,
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
    QKVFlashAttention,
    QKVAttentionLegacy,
    QKVAttention,
)

# === EXTRACTED CONDITIONING COMPONENTS FROM RADIO MAP MODEL ===

class PositionEmbeddingSine(nn.Module):
    """
    Positional embedding using sine/cosine functions.
    Extracted from radio map diffusion model.
    """
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
        # x: b, h, w, d
        num_feats = x.shape[3]
        num_pos_feats = num_feats // 2
        mask = th.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device).to(th.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=th.float32)
        x_embed = not_mask.cumsum(2, dtype=th.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = th.arange(num_pos_feats, dtype=th.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = th.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = th.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = th.cat((pos_y, pos_x), dim=3).contiguous()
        return pos


class Mlp(nn.Module):
    """
    MLP layer for attention mechanism.
    Extracted from radio map diffusion model.
    """
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


class BasicAttentionLayer(nn.Module):
    """
    Cross-attention layer between condition and U-Net features.
    Extracted from radio map diffusion model.
    """
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

    def forward(self, x1, x2):  # x1 for q (conditional input), x2 for k,v
        B, C1, H1, W1 = x1.shape
        _, C2, H2, W2 = x2.shape
        
        shortcut = x2 + self.concat_conv(th.cat(
            [F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=True),
             x2], dim=1))
        shortcut = self.gn(shortcut)
        
        # Handle padding for window operations
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
    """
    Cross-attention relation network wrapper.
    Extracted from radio map diffusion model.
    """
    def __init__(self, in_channel1=128, in_channel2=128, nhead=8, layers=3, embed_dim=128, ffn_dim=512,
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
                BasicAttentionLayer(embed_dim=embed_dim, nhead=nhead, ffn_dim=ffn_dim,
                                    window_size1=window_size1, window_size2=window_size2, dropout=0.1)
            )

    def forward(self, cond, feat):
        cond = self.input_conv1(cond)
        feat = self.input_conv2(feat)
        for att in self.attentions:
            feat = att(cond, feat)
        return feat


# === CONDITIONAL UNET MODEL ===

class ConditionalUNetModel(nn.Module):
    """
    Conditional UNet model that adds cross-attention conditioning to the standard CM UNet.
    Based on the radio map diffusion model architecture.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # NEW CONDITIONING PARAMETERS
        cond_in_dim=3,
        cond_dim=128,
        cond_net="swin",
        use_conditioning=False,
        window_sizes1=None,
        window_sizes2=None,
        fix_backbone=True,
        without_pretrain=False,
    ):
        super().__init__()
        
        # Store conditioning parameters
        self.use_conditioning = use_conditioning
        self.cond_in_dim = cond_in_dim
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # === CONDITIONING COMPONENTS ===
        if self.use_conditioning:
            self.init_conditioning_components(
                cond_net, cond_dim, model_channels, channel_mult,
                window_sizes1, window_sizes2, fix_backbone, without_pretrain
            )

        # Calculate channel dimensions
        ch = input_ch = int(channel_mult[0] * model_channels)
        
        # Input layer - modified for conditioning
        if self.use_conditioning:
            # Add space for condition features
            condition_channels = self.get_condition_channels(cond_net)
            self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(conv_nd(dims, in_channels + condition_channels, ch, 3, padding=1))
            ])
        else:
            self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
            ])
            
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        # Build downsampling blocks
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Build upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # Output layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def init_conditioning_components(self, cond_net, cond_dim, model_channels, channel_mult, 
                                   window_sizes1, window_sizes2, fix_backbone, without_pretrain):
        """Initialize conditioning components based on the radio map model."""
        # For now, we'll create a placeholder condition encoder
        # In a real implementation, you'd import the actual Swin Transformer
        print(f"Initializing {cond_net} condition encoder with {self.cond_in_dim} input channels")
        
        # Placeholder condition encoder (replace with actual Swin Transformer)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(self.cond_in_dim, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Cross-attention layers for each resolution
        dims = [int(model_channels * mult) for mult in channel_mult]
        self.cross_attention_layers = nn.ModuleList()
        
        if window_sizes1 is None:
            window_sizes1 = [[4, 4]] * len(dims)
        if window_sizes2 is None:
            window_sizes2 = [[1, 1]] * len(dims)
            
        for i, dim in enumerate(dims):
            self.cross_attention_layers.append(
                RelationNet(
                    in_channel1=128,  # Condition feature dim
                    in_channel2=dim,  # U-Net feature dim
                    nhead=8,
                    layers=1,
                    embed_dim=dim,
                    ffn_dim=dim * 2,
                    window_size1=window_sizes1[i] if i < len(window_sizes1) else [4, 4],
                    window_size2=window_sizes2[i] if i < len(window_sizes2) else [1, 1]
                )
            )

    def get_condition_channels(self, cond_net):
        """Get the number of channels from condition encoder output."""
        # This should match the actual condition encoder output
        # For placeholder, returning 128
        return 128

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, cond=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param cond: an [N x cond_channels x ...] Tensor of conditions.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        if self.use_conditioning:
            assert cond is not None, "must specify cond if conditioning is enabled"
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        
        # Process conditioning if enabled
        if self.use_conditioning and cond is not None:
            # Extract condition features
            cond_features = self.condition_encoder(cond)
            # Concatenate with input
            h = th.cat([h, F.interpolate(cond_features, size=h.shape[-2:], mode="bilinear")], dim=1)

        # Downsampling
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
            
        # Middle
        h = self.middle_block(h, emb)
        
        # Upsampling  
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
        h = h.type(x.dtype)
        return self.out(h)
