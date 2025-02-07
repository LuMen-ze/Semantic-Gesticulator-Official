import torch as t
import torch.nn as nn 
import numpy as np
from .resnet import Resnet1D
from .Utils.torch_utils import assert_shape

# --------------------------------------------------------
# Reference: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100) -> None:
        super().__init__()

        assert d_model % 2 == 0

        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(t.zeros(1, max_len, d_model), requires_grad=False)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # 编码时同时对【序列位置】以及【特征维度】进行编码
        self.pe.data[0, :, 0::2].copy_(t.sin(position * div_term))
        self.pe.data[0, :, 1::2].copy_(t.cos(position * div_term))

    def forward(self, x) -> t.Tensor:
        """
        x: [N, L, D]
        """
        # print(x.shape, self.pe[:, :x.shape[1], :].shape)
        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)
    
class Transformer_block(nn.Module):
    def __init__(self, embed_dim, lxm_dim, num_heads, depth, dropout=0.1, sample_length=120) -> None:
        super().__init__()
        self.pos_embed = PositionalEncoding(embed_dim, dropout, max_len=sample_length)  # hack: max len of position is 100

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                        nhead=num_heads,
                                                                        dim_feedforward=int(4*embed_dim),
                                                                        dropout=dropout,
                                                                        activation='gelu',
                                                                        batch_first=True),
                                             num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.lxm_embed = nn.Linear(embed_dim, lxm_dim, bias=True)
        self.apply(self._init_weights)
    
    def initialize_weights(self):
        # initialization
        # initialize nn.Parameter
        # t.nn.init.normal_(self.cls_token, std=.02)
        t.nn.init.normal_(self.pos_embed.pe, std=.02)
        # t.nn.init.normal_(self.mask_token, std=.02)
        # t.nn.init.normal_(self.decoder_pos_embed.pe, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            t.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)
            
    def forward(self, x) -> t.Tensor:
        """
        x: [N, L, D]
        """
        x = self.pos_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.lxm_embed(x)
        return x

class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False):
        super().__init__()
        blocks = []
        if stride_t == 1:
            filter_t = 3
            pad_t = 1
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    # nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t, padding_mode='replicate'),
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            # block = nn.Conv1d(width, output_emb_width, 3, 1, 1, padding_mode='replicate')
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None,
                 zero_out=False, res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            # filter_t, pad_t = stride_t * 2, stride_t // 2
            if stride_t == 1:
                filter_t = 3
                pad_t = 1
            else:
                filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            # block = nn.Conv1d(output_emb_width, width, 3, 1, 1, padding_mode='replicate')
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res),
                    nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t)
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width if level == 0 else output_emb_width,
                                                           output_emb_width,
                                                           down_t, stride_t,
                                                           **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs

class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width,
                                                          output_emb_width,
                                                          down_t, stride_t,
                                                          **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)
        # self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1, padding_mode='replicate')
    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
