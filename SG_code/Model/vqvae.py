import numpy as np
import torch as t
import torch.nn as nn

from .encdec import Encoder, Decoder, assert_shape, Transformer_block
from .bottleneck import NoBottleneck, Bottleneck
from .residual_vq import ResidualVQ
from .Utils.logger import average_metrics

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target))


class VQVAE(nn.Module):
    def __init__(self, hps, input_dim):
        super().__init__()
        self.hps = hps  # 模型结构参数。

        input_shape = (hps.sample_length, input_dim)  # 4s(240帧) x 3J.
        levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        l_bins = hps.l_bins  # Codebook size 512.
        mu = hps.l_mu
        commit = hps.commit
        multipliers = hps.hvqvae_multipliers
        self.use_bottleneck = hps.use_bottleneck
        self.bottleneck_type = hps.bottleneck_type
        assert(self.bottleneck_type in ['vqvae', 'rqvae'])
        if self.use_bottleneck:
            print('We use bottleneck!, the type is', self.bottleneck_type)
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                            dilation_growth_rate=hps.dilation_growth_rate,
                            dilation_cycle=hps.dilation_cycle,
                            reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]  # (240,) 3J
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)  # [8]
        self.hop_lengths = np.cumprod(self.downsamples)  # [8]
        self.z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]  # [30]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        self.has_encoder_transformer = hasattr(hps, 'encoder_transformer') and hps.encoder_transformer
        self.has_decoder_transformer = hasattr(hps, 'decoder_transformer') and hps.decoder_transformer
        self.has_after_decoder_transformer = hasattr(hps, 'has_after_decoder_transformer') and hps.has_after_decoder_transformer
        if self.has_encoder_transformer:
            self.encoder_transformer = Transformer_block(embed_dim=hps.emb_width, lxm_dim=hps.emb_width, num_heads=hps.transformer_head, depth=hps.transformer_depth, dropout=hps.transformer_dropout)
        if self.has_decoder_transformer:
            self.decoder_transformer = Transformer_block(embed_dim=hps.emb_width, lxm_dim=hps.emb_width, num_heads=hps.transformer_head, depth=hps.transformer_depth, dropout=hps.transformer_dropout)
        if self.has_after_decoder_transformer:
            self.after_decoder_transformer = Transformer_block(embed_dim=hps.emb_width, lxm_dim=x_channels, num_heads=hps.after_d_transformer_head, depth=hps.after_d_transformer_depth, dropout=hps.after_d_transformer_dropout)

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        if self.has_after_decoder_transformer:
            decoder = lambda level: Decoder(emb_width, emb_width, level + 1,
                                            downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        else:
            decoder = lambda level: Decoder(x_channels, emb_width, level + 1,
                                            downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        self.norm_bottleneck = hasattr(hps, 'norm_bottleneck') and hps.norm_bottleneck
        if self.norm_bottleneck:
            self.norm_bottleneck_layer = nn.BatchNorm1d(emb_width)

        if self.use_bottleneck and self.bottleneck_type == 'vqvae':
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        elif self.use_bottleneck and self.bottleneck_type == 'rqvae':
            self.bottleneck = ResidualVQ(
            dim = hps.emb_width,
            num_quantizers = hps.rq_num_quantizers,      # specify number of quantizers
            codebook_size = l_bins,    # codebook size
            decay = mu,            # decay for moving averages (default: 0.95)
            commitment_weight = commit, # weight of commitment loss (default: 0.02)
        )
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0
        self.acc = hps.acc if hasattr(hps, 'acc') else 0
        self.vel = hps.vel if hasattr(hps, 'vel') else 0
        self.fore_foot = hps.fore_foot if hasattr(hps, 'fore_foot') else 0
        if self.reg is 0:
            print('No motion regularization!')

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None, insert_pos=None, stride=1, filter_name='savgol', filter_pos=None, skip_botleneck=False):
        # Decode
        if end_level is None:
            end_level = self.levels
        if start_level is None:
            start_level = 0

        if not skip_botleneck:
            assert len(zs) == end_level - start_level
            if self.bottleneck_type == 'rqvae':
                xs_quantised = self.bottleneck.decode(zs[0][0])
                xs_quantised = [xs_quantised.unsqueeze(0).permute(0, 2, 1)]
            elif self.bottleneck_type == 'vqvae':
                xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)     
                xs_quantised[0] = xs_quantised[0].permute(0, 2, 1)     
            else:
                raise NotImplementedError
        else:
            xs_quantised = zs
            xs_quantised[0] = xs_quantised[0].permute(0, 2, 1)
            print("++++", xs_quantised[0].shape)

        if filter_pos is not None:
            assert filter_pos in ['motion_info', 'rq_code']
        if filter_pos == 'rq_code':
            # xs_quantised[0]: [1, D, L]
            np_data = xs_quantised[0].cpu().numpy()
            if filter_name == 'savgol':
                smoothed_np_data = savgol_filter(np_data[0].transpose(1,0), window_length=5, polyorder=2, axis=0)
            elif filter_name == 'gaussian':
                np_data = np_data.transpose(0, 2, 1)
                smoothed_np_data = np.zeros_like(np_data[0])
                dim = smoothed_np_data.shape[-1]
                sigma = 2
                for i in range(dim):
                    smoothed_np_data[:, i] = gaussian_filter1d(np_data[0, :, i], sigma)

            xs_quantised[0] = t.from_numpy(smoothed_np_data).permute(1,0).unsqueeze(0).to(zs[0][0].device)
            # print(xs_quantised[0].shape)

        assert len(xs_quantised) == end_level - start_level

        if self.has_decoder_transformer:
            # xs[0]: [N, D, L], x_before_decoder_trans: [N, L, D]
            x_before_decoder_trans = xs_quantised[0].permute(0, 2, 1)
            assert(x_before_decoder_trans.shape[0] == 1)

            out_n, out_l, out_d = x_before_decoder_trans.shape
            stride = stride
            # x_before_decoder_trans: [1, L, D] -> [x, 15, D]
            if out_l > 15:
                last_window = x_before_decoder_trans[:, -15:, :]
                # window_size: 15, stride: 1
                x_before_decoder_trans_fold = x_before_decoder_trans.unfold(1, 15, stride).squeeze(0).permute(0, 2, 1)
                need_add_last_frames = ((out_l - 15) % stride != 0)
                if need_add_last_frames:
                    x_before_decoder_trans_fold = t.cat((x_before_decoder_trans_fold, last_window), dim=0)
                
                if insert_pos is not None:
                    sg_code_window = None
                    for i in range(len(insert_pos)):
                        s_i, e_i = insert_pos[i]
                        if e_i <= 15:
                            sg_window = x_before_decoder_trans[:, 0:15, :]
                        else:
                            sg_window = x_before_decoder_trans[:, e_i-15:e_i, :]
                        if sg_code_window is None:
                            sg_code_window = sg_window
                        else:
                            sg_code_window = t.cat((sg_code_window, sg_window), dim=0)
                    sg_code_after_decoder_trans = self.decoder_transformer(sg_code_window)
            else:
                x_before_decoder_trans_fold = x_before_decoder_trans
            x_after_decoder_trans = self.decoder_transformer(x_before_decoder_trans_fold)

            if(x_after_decoder_trans.shape[0] > 1):
                if need_add_last_frames:
                    x_out_last_frame = x_after_decoder_trans[1:-1,-stride:,:].reshape(1, -1, out_d)
                    need_add_last_frames_num = (out_l - 15) % stride
                    print('need_add_last_frames_num', need_add_last_frames_num)
                    x_after_decoder_trans = t.cat((x_after_decoder_trans[0:1,:,:], x_out_last_frame, x_after_decoder_trans[-1:,-need_add_last_frames_num:, :]), dim=1)
                else:
                    x_out_last_frame = x_after_decoder_trans[1:,-stride:,:].reshape(1, -1, out_d)
                    x_after_decoder_trans = t.cat((x_after_decoder_trans[0:1,:,:], x_out_last_frame), dim=1)
                assert(x_after_decoder_trans.shape == (1, out_l, out_d))

                if insert_pos is not None:
                    assert(len(insert_pos) == sg_code_after_decoder_trans.shape[0])
                    for i in range(len(insert_pos)):
                        s_i, e_i = insert_pos[i]
                        if e_i <= 15:
                            assert(i == 0)
                            x_after_decoder_trans[:, 0:15, :] = sg_code_after_decoder_trans[0: 1]
                        else:
                            x_after_decoder_trans[:, e_i-15:e_i, :] = sg_code_after_decoder_trans[i: i+1]
            
            xs_quantised[0] = x_after_decoder_trans.permute(0, 2, 1)

        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)


        if self.has_after_decoder_transformer:
            print("has after decoder t")
            # xs[0]: [N, D, L], x_before_decoder_trans: [N, L, D]
            x_before_conv_decoder_trans = x_out.permute(0,2,1)
            out_n, out_l, out_d = x_before_conv_decoder_trans.shape
            if out_l > 120:
                # window_size: 120, stride: 1
                x_before_conv_decoder_trans = x_before_conv_decoder_trans.unfold(1, 120, 1).squeeze(0).permute(0, 2, 1)

            x_after_conv_decoder_trans = self.after_decoder_transformer(x_before_conv_decoder_trans)
            out_d = x_after_conv_decoder_trans.shape[-1]
            
            if(x_after_conv_decoder_trans.shape[0] > 1):
                x_out_last_frame = x_after_conv_decoder_trans[1:,-1:,:].reshape(1, -1, out_d)
                x_after_conv_decoder_trans = t.cat((x_after_conv_decoder_trans[0:1,:,:], x_out_last_frame), dim=1)
                assert(x_after_conv_decoder_trans.shape == (1, out_l, out_d))
            
            x_out = x_after_conv_decoder_trans.permute(0, 2, 1)
        
        x_out = self.postprocess(x_out)

        # filter_pos = 'motion_info'
        # filter_name = 'gaussian'
        if filter_pos == 'motion_info':
            # x_out: [1, L, D]
            np_data = x_out.detach().cpu().numpy()
            if filter_name == 'savgol':
                smoothed_np_data = savgol_filter(np_data[0], window_length=24, polyorder=4, axis=0)
            elif filter_name == 'gaussian':
                smoothed_np_data = np.zeros_like(np_data[0])
                dim = smoothed_np_data.shape[-1]
                sigma = 5
                for i in range(dim):
                    smoothed_np_data[:, i] = gaussian_filter1d(np_data[0, :, i], sigma)

            x_out = t.from_numpy(smoothed_np_data).unsqueeze(0).to(zs[0][0].device)

        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1, insert_pos=None, stride=1, filter_name='savgol', filter_pos=None, skip_botleneck=False):
        # print(zs.shape)
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level, insert_pos=insert_pos, stride=stride, filter_name=filter_name, filter_pos=filter_pos, skip_botleneck=skip_botleneck)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        # print('xs', xs[0])
        assert(len(xs) == 1)

        if self.has_encoder_transformer:
            # print('xs', xs[0].shape)
            assert(xs[0].shape[0] == 1 or xs[0].shape[2] <= 15)
            # xs[0]: [N, D, L], x_before_trans: [N, L, D]
            x_before_trans = xs[0].permute(0,2,1)
            out_n, out_l, out_d = x_before_trans.shape

            need_slide_window = False
            if out_l > 15:
                # window_size: 15, stride: 1
                x_before_trans = x_before_trans.unfold(1, 15, 1).squeeze(0).permute(0, 2, 1)
                # print(x_before_trans.shape)
                need_slide_window = True

            x_after_trans = self.encoder_transformer(x_before_trans)
            
            if(need_slide_window):
                x_out_last_frame = x_after_trans[1:,-1:,:].reshape(1, -1, out_d)
                x_after_trans = t.cat((x_after_trans[0:1,:,:], x_out_last_frame), dim=1)
                print(x_after_trans.shape, out_n, out_l, out_d)
                assert(x_after_trans.shape == (1, out_l, out_d))

            xs[0] = x_after_trans.permute(0, 2, 1)

        if self.norm_bottleneck:
            xs[0] = self.norm_bottleneck_layer(xs[0])

        if self.bottleneck_type == 'rqvae':
            zs = self.bottleneck.encode(xs[0])
            # print('zs', len(zs), zs[0].shape)
            return t.stack(zs)
        else:
            zs = self.bottleneck.encode(xs)
            return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            # print('zs_i', zs_i.shape)
            zs_list.append(zs_i)
        # zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        assert(len(x_chunks) == 1)
        return zs_list[0] # [4, bs, len]
    
    def one_hot_decode(self, one_hot, type='one_hot'):
        if type == 'normal':
            xs_quantised = self.bottleneck.decode_for_1_layer(one_hot)
        elif type == 'one_hot':
            xs_quantised = self.bottleneck.one_hot_decode(one_hot)
        else:
            raise NotImplementedError

        if self.has_decoder_transformer:
            x_after_decoder_trans = self.decoder_transformer(xs_quantised)
            xs_quantised = x_after_decoder_trans.permute(0, 2, 1)
            
        x_outs = []
        assert(self.levels == 1)
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder([xs_quantised], all_levels=False)
            x_outs.append(x_out)

        if self.has_after_decoder_transformer:
            x_after_conv_decoder_trans = self.after_decoder_transformer(x_outs[0].permute(0, 2, 1))
            x_outs[0] = x_after_conv_decoder_trans.permute(0, 2, 1)

        return x_outs[0].permute(0, 2, 1)


    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def inference(self, x, out_encode_value=False):
        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        assert(len(xs) == 1)
        if self.has_encoder_transformer:
            assert(xs[0].shape[0] == 1)
            # xs[0]: [N, D, L], x_before_trans: [N, L, D]
            x_before_trans = xs[0].permute(0,2,1)
            out_n, out_l, out_d = x_before_trans.shape
            # window_size: 15, stride: 1
            if out_l > 15:
                x_before_trans = x_before_trans.unfold(1, 15, 1).squeeze(0).permute(0, 2, 1)
                print(x_before_trans.shape)

            x_after_trans = self.encoder_transformer(x_before_trans)
            
            if(x_after_trans.shape[0] > 1):
                x_out_last_frame = x_after_trans[1:,-1:,:].reshape(1, -1, out_d)
                x_after_trans = t.cat((x_after_trans[0:1,:,:], x_out_last_frame), dim=1)
                print(x_after_trans.shape, out_n, out_l, out_d)
                assert(x_after_trans.shape == (1, out_l, out_d))

            xs[0] = x_after_trans.permute(0, 2, 1)
            # assert(0)

        if self.norm_bottleneck:   
            xs[0] = self.norm_bottleneck_layer(xs[0])
        if self.bottleneck_type == 'rqvae':
            xs_quantised, zs, _, _, _ = self.bottleneck(xs[0])
            xs_quantised = [xs_quantised]
            zs = [zs]
        else:
            zs, xs_quantised, _, _ = self.bottleneck(xs)

        if self.has_decoder_transformer:
            # xs[0]: [N, D, L], x_before_decoder_trans: [N, L, D]
            x_before_decoder_trans = xs_quantised[0].permute(0,2,1)
            out_n, out_l, out_d = x_before_decoder_trans.shape
            if out_l > 15:
                # window_size: 15, stride: 1
                x_before_decoder_trans = x_before_decoder_trans.unfold(1, 15, 1).squeeze(0).permute(0, 2, 1)

            x_after_decoder_trans = self.decoder_transformer(x_before_decoder_trans)

            if(x_after_decoder_trans.shape[0] > 1):
                x_out_last_frame = x_after_decoder_trans[1:,-1:,:].reshape(1, -1, out_d)
                x_after_decoder_trans = t.cat((x_after_decoder_trans[0:1,:,:], x_out_last_frame), dim=1)
                assert(x_after_decoder_trans.shape == (1, out_l, out_d))
            
            xs_quantised[0] = x_after_decoder_trans.permute(0, 2, 1)
            
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            if not self.has_after_decoder_transformer:
                assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        if self.has_after_decoder_transformer:
            print("has after decoder t")
            # xs[0]: [N, D, L], x_before_decoder_trans: [N, L, D]
            x_before_conv_decoder_trans = x_outs[0].permute(0,2,1)
            out_n, out_l, out_d = x_before_conv_decoder_trans.shape
            stride = 8
            if out_l > 120:
                last_window = x_before_conv_decoder_trans[:, -120:, :]
                # window_size: 120, stride: 1
                x_before_conv_decoder_trans = x_before_conv_decoder_trans.unfold(1, 120, stride).squeeze(0).permute(0, 2, 1)
                need_add_last_frames = ((out_l - 120) % stride != 0)
                if need_add_last_frames:
                    x_before_conv_decoder_trans = t.cat((x_before_conv_decoder_trans, last_window), dim=0)

            x_after_conv_decoder_trans = self.after_decoder_transformer(x_before_conv_decoder_trans)
            out_d = x_after_conv_decoder_trans.shape[-1]
            
            if(x_after_conv_decoder_trans.shape[0] > 1):
                if need_add_last_frames:
                    x_out_last_frame = x_after_conv_decoder_trans[1:-1,-stride:,:].reshape(1, -1, out_d)
                    need_add_last_frames_num = (out_l - 120) % stride
                    x_after_conv_decoder_trans = t.cat((x_after_conv_decoder_trans[0:1,:,:], x_out_last_frame, x_after_conv_decoder_trans[-1:,-need_add_last_frames_num:, :]), dim=1)
                else:
                    x_out_last_frame = x_after_conv_decoder_trans[1:,-stride:,:].reshape(1, -1, out_d)
                    x_after_conv_decoder_trans = t.cat((x_after_conv_decoder_trans[0:1,:,:], x_out_last_frame), dim=1)
                assert(x_after_conv_decoder_trans.shape == (1, out_l, out_d))
            
            x_outs[0] = x_after_conv_decoder_trans.permute(0, 2, 1)

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])

        if out_encode_value:
            return x_out, xs[0]
        else:
            return x_out

    def forward(self, x):
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        assert(len(xs) == 1)
        if self.has_encoder_transformer:
            x_after_trans = self.encoder_transformer(xs[0].permute(0, 2, 1))
            xs[0] = x_after_trans.permute(0, 2, 1)

        if self.norm_bottleneck:
            xs[0] = self.norm_bottleneck_layer(xs[0])

        if self.use_bottleneck == True and self.bottleneck_type == 'rqvae':
            xs_quantised, zs, sep_commit_losses, final_commit_loss, quantiser_metrics = self.bottleneck(xs[0])
            xs_quantised = [xs_quantised]
            zs = [zs]
            commit_losses = [t.sum(sep_commit_losses)]
        else:
            zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)

        if self.has_decoder_transformer:
            x_after_decoder_trans = self.decoder_transformer(xs_quantised[0].permute(0, 2, 1))
            xs_quantised[0] = x_after_decoder_trans.permute(0, 2, 1)
            
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            if not self.has_after_decoder_transformer:
                assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        if self.has_after_decoder_transformer:
            x_after_conv_decoder_trans = self.after_decoder_transformer(x_outs[0].permute(0, 2, 1))
            assert_shape(x_after_conv_decoder_trans.permute(0, 2, 1), x_in.shape)
            x_outs[0] = x_after_conv_decoder_trans.permute(0, 2, 1)

        recons_loss = t.zeros(()).to(x.device)
        regularization = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)
        x_target = x.float()
        fore_foot_recons_loss = t.zeros(()).to(x.device)

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])

            this_recons_loss = _loss_fn(x_target, x_out)

            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss

            recons_loss += this_recons_loss
            regularization += t.mean((x_out[:, 1:] - x_out[:, :-1]) ** 2) # 约束速度不要太大

            velocity_loss += _loss_fn(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            acceleration_loss += _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1],
                                          x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        commit_loss = sum(commit_losses).to(x.device)


        if self.use_bottleneck == True and self.bottleneck_type == 'rqvae':
            loss = recons_loss + commit_loss + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss
            quantiser_metrics = average_metrics(quantiser_metrics)
        else:
            loss = recons_loss + commit_loss * self.commit + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss
            quantiser_metrics = average_metrics(quantiser_metrics)

        with t.no_grad():
            l1_loss = _loss_fn(x_target, x_out)

        if self.use_bottleneck == True and self.bottleneck_type == 'rqvae':
            metrics.update(dict(
                recons_loss=recons_loss,
                l1_loss=l1_loss,
                commit_loss=commit_loss,
                sep_commit_loss_mean=t.mean(sep_commit_losses),
                regularization=regularization,
                velocity_loss=velocity_loss,
                acceleration_loss=acceleration_loss,
                **quantiser_metrics))
        else:
            metrics.update(dict(
                recons_loss=recons_loss,
                l1_loss=l1_loss,
                commit_loss=commit_loss,
                sep_commit_loss_mean=t.tensor(0),
                regularization=regularization,
                velocity_loss=velocity_loss,
                acceleration_loss=acceleration_loss,
                **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics
    

    def forward_skip_bottleneck(self, x):

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        # ! skip the bottleneck and use xs directly
        # zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs[level:level + 1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)


        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])

        return x_out       