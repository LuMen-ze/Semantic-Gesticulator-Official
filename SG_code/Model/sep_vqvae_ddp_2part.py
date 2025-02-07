# region Import

import torch
import torch.nn as nn
import numpy as np
import copy

from .vqvae import VQVAE

# endregion
def _loss_fn(x_target, x_pred):
    return torch.mean(torch.abs(x_pred - x_target))

class SepVQVAE_DDP_body_hands(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.channel_num = hps.joint_channel
        self.joint_idxes_body = hps.body.joint_indexes
        self.joint_idxes_hands = hps.hands.joint_indexes

        self.joint_num = hps.joint_num

        self.hands_joint_dim_selected = np.arange(0,3*len(self.joint_idxes_hands))

        self.vqvae_body = VQVAE(hps.body, 3*len(self.joint_idxes_body))
        self.vqvae_hands = VQVAE(hps.hands, len(self.hands_joint_dim_selected))

        self.body_weight = hps.body.weight
        self.hands_weight = hps.hands.weight


    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1, insert_pos=None, stride=1, filter_name='savgol', filter_pos=None, skip_botleneck=False):
        """
        zs are list with three elements: z for body, z for down and z for face
        """
        z_body = zs[0]
        z_hands = zs[1]

        x_body = self.vqvae_body.decode(z_body, insert_pos=insert_pos, stride=stride, filter_name=filter_name, filter_pos=filter_pos, skip_botleneck=skip_botleneck)
        x_hands = self.vqvae_hands.decode(z_hands, insert_pos=insert_pos, stride=stride, filter_name=filter_name, filter_pos=filter_pos, skip_botleneck=skip_botleneck)

        b, t, _ = x_body.shape

        x_out_body =  torch.zeros(b, t, len(self.joint_idxes_body) * self.channel_num).cuda().float()
        x_out_hands = torch.zeros(b, t, len(self.joint_idxes_hands) * self.channel_num).cuda().float()

        x_out_body = x_body
        x_out_hands[:, :, self.hands_joint_dim_selected] = x_hands

        x_out_body_all = torch.zeros(b, t, self.joint_num+1, self.channel_num).cuda().float()
        x_out_body_all[:, :, self.joint_idxes_body, :] = x_out_body.view(b, t, len(self.joint_idxes_body), self.channel_num)
        x_out_body_all[:, :, self.joint_idxes_hands, :] = x_out_hands.view(b, t, len(self.joint_idxes_hands), self.channel_num)

        return x_out_body_all.view(b, t, -1)

    def encode(self, x_all_body, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x_all_body.size()
        x_body = x_all_body.view(b, t, c//self.channel_num, self.channel_num)[:, :, self.joint_idxes_body].view(b, t, -1)

        x_hands_mid = x_all_body.view(b, t, c//self.channel_num, self.channel_num)[:, :, self.joint_idxes_hands].view(b, t, -1)
        x_hands = x_hands_mid[:, :, self.hands_joint_dim_selected]
        
        z_body = self.vqvae_body.encode(x_body, start_level, end_level, bs_chunks)
        z_hands = self.vqvae_hands.encode(x_hands, start_level, end_level, bs_chunks)

        # print('z_body', z_body.shape) # [4, bs, len]
        return (z_body, z_hands)

    def sample(self, n_samples):
        x_body = self.vqvae_body.sample(n_samples)

        raise NotImplementedError
 
    def inference(self, x_all_body, out_encode_value=False):
        b, t, c = x_all_body.size()
        x_all_body = x_all_body.view(b, t, c//self.channel_num, self.channel_num)
        _, _, j, _ = x_all_body.size()
        now_device = x_all_body.device

        x_body = x_all_body[:, :, self.joint_idxes_body, :].view(b, t, -1)

        x_hands_mid = x_all_body[:, :, self.joint_idxes_hands, :].view(b, t, -1)
        x_hands = x_hands_mid[:, :, self.hands_joint_dim_selected]
        
        if out_encode_value:
            x_out_body, x_encode_value_body = self.vqvae_body.inference(x_body, out_encode_value=True)
            x_out_hands, x_encode_value_hands = self.vqvae_hands.inference(x_hands, out_encode_value=True)
        else:
            x_out_body = self.vqvae_body.inference(x_body)
            x_out_hands = self.vqvae_hands.inference(x_hands)

        x_out_body_mid =  torch.zeros(b, t, len(self.joint_idxes_body) * self.channel_num).to(now_device).float()
        x_out_hands_mid = torch.zeros(b, t, len(self.joint_idxes_hands) * self.channel_num).to(now_device).float()
        
        x_out_body_mid = x_out_body
        x_out_hands_mid[:, :, self.hands_joint_dim_selected] = x_out_hands

        x_out_body = torch.zeros(b, t, j, self.channel_num).to(now_device).float()
        x_out_body[:, :, self.joint_idxes_body, :] = x_out_body_mid.view(b, t, len(self.joint_idxes_body), self.channel_num)
        x_out_body[:, :, self.joint_idxes_hands, :] = x_out_hands_mid.view(b, t, len(self.joint_idxes_hands), self.channel_num)

        if out_encode_value:
            return x_out_body.view(b, t, -1), [x_encode_value_body, x_encode_value_hands]
        else:
            return x_out_body.view(b, t, -1)
    

    def forward(self, x_all_body):
        b, t, c = x_all_body.size()
        x_all_body = x_all_body.view(b, t, c//self.channel_num, self.channel_num)
        _, _, j, _ = x_all_body.size()
        now_device = x_all_body.device

        x_body = x_all_body[:, :, self.joint_idxes_body, :].view(b, t, -1)

        x_hands_mid = x_all_body[:, :, self.joint_idxes_hands, :].view(b, t, -1)
        x_hands = x_hands_mid[:, :, self.hands_joint_dim_selected]
    
        x_out_body, loss_body, metrics_body = self.vqvae_body(x_body)
        x_out_hands, loss_hands, metrics_hands = self.vqvae_hands(x_hands)

        x_out_body_mid =  torch.zeros(b, t, len(self.joint_idxes_body) * self.channel_num).to(now_device).float()
        x_out_hands_mid = torch.zeros(b, t, len(self.joint_idxes_hands) * self.channel_num).to(now_device).float()
        
        x_out_body_mid = x_out_body
        x_out_hands_mid[:, :, self.hands_joint_dim_selected] = x_out_hands

        x_out_body = torch.zeros(b, t, j, self.channel_num).to(now_device).float()
        x_out_body[:, :, self.joint_idxes_body, :] = x_out_body_mid.view(b, t, len(self.joint_idxes_body), self.channel_num)
        x_out_body[:, :, self.joint_idxes_hands, :] = x_out_hands_mid.view(b, t, len(self.joint_idxes_hands), self.channel_num)

        return x_out_body.view(b, t, -1), self.body_weight * loss_body + self.hands_weight * loss_hands, loss_body, loss_hands, [metrics_body, metrics_hands]
    

    def forward_skip_bottleneck(self, x_body):
        b, t, c = x_body.size()
        x_body = x_body.view(b, t, c//self.channel_num, self.channel_num)
        _, _, j, _ = x_body.size()
        now_device = x_body.device

        x_body = x_body[:, :, self.joint_idxes_body, :].view(b, t, -1)


        x_hands_mid = x_body[:, :, self.joint_idxes_hands, :].view(b, t, -1)
        x_hands = x_hands_mid[:, :, self.hands_joint_dim_selected]
        
        x_out_body = self.vqvae_body.forward_skip_bottleneck(x_body)
        x_out_hands = self.vqvae_hands.forward_skip_bottleneck(x_hands)

        x_out_body_mid =  torch.zeros(b, t, len(self.joint_idxes_body) * self.channel_num).to(now_device).float()
        x_out_hands_mid = torch.zeros(b, t, len(self.joint_idxes_hands) * self.channel_num).to(now_device).float()
        
        x_out_body_mid = x_out_body
        x_out_hands_mid[:, :, self.hands_joint_dim_selected] = x_out_hands

        x_out_body = torch.zeros(b, t, j, self.channel_num).to(now_device).float()
        x_out_body[:, :, self.joint_idxes_body, :] = x_out_body_mid.view(b, t, len(self.joint_idxes_body), self.channel_num)
        x_out_body[:, :, self.joint_idxes_hands, :] = x_out_hands_mid.view(b, t, len(self.joint_idxes_hands), self.channel_num)

        return x_out_body.view(b, t, -1)