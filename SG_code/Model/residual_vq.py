import random
from math import ceil
from functools import partial
from itertools import zip_longest
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from .bottleneck import NoBottleneck, Bottleneck, BottleneckBlock

from einops import rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        dim, # input dimension
        codebook_size, # codebook size
        num_quantizers, # rq中codebook数量
        codebook_dim = None, # codebook dimension，这里assert codebook_dim == dim，所以输入时不需要指定codebook_dim
        shared_codebook = False, # 所有层共享同一个 codebook
        heads = 1, # heads == 1
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        decay = 0.95,
        commitment_weight = 0.02,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, 'residual vq is not compatible with multi-headed codes'
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight

        self.accept_image_fmap = accept_image_fmap
        self.layers = nn.ModuleList([BottleneckBlock(k_bins = codebook_size, emb_width = codebook_dim, mu = decay) for _ in range(num_quantizers)])

        # assert all([not vq.has_projections for vq in self.layers])

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        codebooks = rearrange(codebooks, 'q 1 c d -> q c d')
        return codebooks

    # 输入rq各层的index，输出每层得到的code
    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    # 将上一个函数中每层得到的code加起来得到最终的code
    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def encode(self, x):
        residual = x
        all_indices = []
        # print('rvq-encode')
        for quantizer_index, layer in enumerate(self.layers):
            # quantized, *rest = layer(residual)
            # print(residual.shape)
            embed_indices, quantized, commit_loss, quantiser_metrics = layer(residual, update_k=False)
            all_indices.append(embed_indices)
            residual = residual - quantized.detach()
            # if quantizer_index == 0:
            #     print(embed_indices)
        
        # all_indices = torch.stack(all_indices, dim=-1)
        # print(all_indices.shape)
        return all_indices

    def decode(self, indices):
        print(indices.shape)
        quantized_out = None
        for quantizer_index, layer in enumerate(self.layers):
            layer_quantized_out = layer.dequantise(indices[quantizer_index])
            if quantized_out is None:
                quantized_out = layer_quantized_out
            else:
                quantized_out += layer_quantized_out
        return quantized_out
    
    def decode_for_1_layer(self, indices):
        layer_quantized_out = self.layers[0].dequantise(indices)
        return layer_quantized_out
    
    def one_hot_decode(self, one_hot):
        # one_hot: [batch, seq, dim]
        layer_quantized_out = self.layers[0].one_hot_decode(one_hot)
        return layer_quantized_out

    def forward(
        self,
        x,
        indices = None,
        return_all_codes = False, # 如果设为true，则返回值多一个all_codes，包括每一个vq层经过对应codebook得到的量化值
        sample_codebook_temp = None,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, return_loss, device = self.num_quantizers, self.quantize_dropout_multiple_of, exists(indices), x.device

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        # 最终的输出
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(indices == -1), 'some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss'
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:
            rand = random.Random(rand_quantize_dropout_fixed_seed) if exists(rand_quantize_dropout_fixed_seed) else random

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            null_loss = torch.full((1,), 0., device = device, dtype = x.dtype)

        # go through the layers
        quantiser_metricses = []
        for quantizer_index, layer in enumerate(self.layers):
            # # ! debug 
            # if quantizer_index >= 4:
            #     break

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            # quantized, *rest = layer(residual)
            # print(residual.shape)
            embed_indices, quantized, commit_loss, quantiser_metrics = layer(residual, update_k=self.training)
            loss = commit_loss * self.commitment_weight
            quantiser_metricses.append(quantiser_metrics)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            # if return_loss:
            #     ce_loss = rest[0]
            #     ce_losses.append(ce_loss)
            #     continue

            # embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # quantized_out = torch.einsum('NDL->NLD', quantized_out)

        # final_commit_loss = torch.norm(quantized_out.detach() - x) ** 2 / np.prod(x.shape)
        # final_commit_loss = final_commit_loss * self.commitment_weight
        # project out, if needed
        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))
        final_commit_loss = torch.sum(all_losses)

        # quantized_out: (batch, seq, dim)
        # all_indices: (quantizer, batch, seq)
        # all_losses: (quantizer, batch) 在Image Generation RVQ 这篇paper中，对于commitment loss的处理方法是对每个rq层上的loss取平均值
        ret = (quantized_out, all_indices, all_losses, final_commit_loss, quantiser_metricses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret


if __name__ == "__main__":

    residual_vq = ResidualVQ(
        dim = 512,
        num_quantizers = 4,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
        decay = 0.95,            # decay for moving averages (default: 0.95)
        commitment_weight = 0.02, # weight of commitment loss (default: 0.02)
        # use_cosine_sim = True,
    )

    x = torch.randn(256, 120, 512)

    quantized, indices, commit_loss = residual_vq(x)
    print(quantized.shape, indices.shape, commit_loss.shape)
    print(commit_loss)
