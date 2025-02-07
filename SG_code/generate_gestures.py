# region Import

import os
import json
import torch
import Model
import argparse
import shutil
import copy

import numpy as np
import torch.distributed as dist

from easydict import EasyDict
from prepare_data import Processor
from Utils.utils import load_standard_scalers, to_bvh

# endregion


def generate(data_dir, audio_name, audio_path, gpt_path, rqvae_path, use_gpu, init_body_pose_code, init_hands_pose_code, local_rank):
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        data_config = EasyDict(json.load(f))

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu', local_rank)

    model_gpt = []
    for i in range(len(gpt_path)):
        gpt_checkpoint = torch.load(gpt_path[i], map_location='cpu')
        gpt_config = EasyDict(gpt_checkpoint['config'])

        model_gpt_i = torch.nn.parallel.DistributedDataParallel(
            getattr(Model, gpt_config.structure_gpt.name)(gpt_config.structure_gpt).cuda(),
            device_ids=[local_rank],
            output_device=local_rank
        )
        model_gpt_i.load_state_dict(gpt_checkpoint['model'])
        model_gpt.append(model_gpt_i)

    rq_checkpoint = torch.load(rqvae_path, map_location='cpu')
    rq_config = EasyDict(rq_checkpoint['config'])
    rq_type = rq_config.structure.name
    model_vqvae = torch.nn.parallel.DistributedDataParallel(
        getattr(Model, rq_config.structure.name)(rq_config.structure).cuda(),
        device_ids=[local_rank],
        output_device=local_rank,
    )
    model_vqvae.load_state_dict(rq_checkpoint['model'])

    with torch.no_grad():
        for model_gpt_i in model_gpt:
            model_gpt_i.eval()
        model_vqvae.eval()

        audio_feats = np.load(os.path.join(data_dir, "audio_features.npy")).astype(np.float32)  # 1, time, feature
        audio_feats = torch.from_numpy(audio_feats).to(device)

        init_z_body = torch.ones(1, 1,).to(device).long() * init_body_pose_code
        init_z_hands = torch.ones(1, 1,).to(device).long() * init_hands_pose_code
        init_z = (init_z_body, init_z_hands)

        # print("init_z is: ", init_z)
        zs = model_gpt[0].module.sample(init_z, cond=audio_feats[:, 1:, :])

        # body_0[0]: [1, len], audio_feats: [1, len, 437]
        body_0, hands_0 = zs

        # body_0: [bs, 15], audio_feats_for_inference:[bs, 15, 437]
        body_0 = body_0[0].unfold(1, 15, 1).squeeze(0)
        hands_0 = hands_0[0].unfold(1, 15, 1).squeeze(0)
        audio_feats_for_inference = audio_feats.unfold(1, 15, 1).permute(0, 1, 3, 2).squeeze(0)
        print("test:", body_0.shape, hands_0.shape, audio_feats_for_inference.shape)

        quants_input = (body_0, hands_0)
        body_code_all = [body_0]
        hands_code_all = [hands_0]

        for i in range(1, len(model_gpt)):
            print(i)
            zs_i, _, _ = model_gpt[i](quants_input, audio_feats_for_inference, None)
            # zs_i[0]: [bs, 15]
            if i == 1:
                quants_input = (quants_input, zs_i)
            else:
                quants_input = quants_input + (zs_i,)
            body_code_all.append(zs_i[0])
            hands_code_all.append(zs_i[1])

        # body_code_all: [4, bs, 15]
        body_code_all = torch.stack(body_code_all, dim=0)
        hands_code_all = torch.stack(hands_code_all, dim=0)

        # body_code_all: [4, bs, 15]->[4, 1, len]
        x_out_last_frame_body = body_code_all[:, 1:,-1:].reshape(4, 1, -1) # [4, bs-1, 1]->[4, 1, bs-1]
        body_code_all = torch.cat((body_code_all[:,0:1,:], x_out_last_frame_body), dim=2)
        assert(torch.equal(body_code_all[0], zs[0][0]) == True)

        x_out_last_frame_hands = hands_code_all[:, 1:,-1:].reshape(4, 1, -1) # [4, bs-1, 1]->[4, 1, bs-1]
        hands_code_all = torch.cat((hands_code_all[:,0:1,:], x_out_last_frame_hands), dim=2)
        assert(torch.equal(hands_code_all[0], zs[1][0]) == True)

        tuple_code = ([body_code_all.permute(1,0,2)], [hands_code_all.permute(1,0,2)])
        generated_result_rq_s1 = model_vqvae.module.decode(tuple_code, stride=1)
        generated_result_rq_s1_filter = model_vqvae.module.decode(tuple_code, stride=1, filter_name='savgol', filter_pos='rq_code')

        need_visulize_motion = [generated_result_rq_s1_filter]
        save_name = ['original_motion']


    
    body_scaler = load_standard_scalers(os.path.join(rq_config.data.dir, "body_scaler.sav"))
    joint_num = gpt_config.structure_vqvae.joint_num

    for i in range(len(need_visulize_motion)):
        # data: [frame, 225(motion feature)]
        data = need_visulize_motion[i].data[0].cpu().numpy()
        print(data.shape)
        if data.shape[1] == joint_num * 3:
            data = np.concatenate((data, np.zeros((data.shape[0], 3))), axis=1)
        data = body_scaler.inverse_transform(data)

        to_bvh(data, gpt_config, os.path.join(data_dir, audio_name+'_'+save_name[i]+'.bvh'))


if __name__ == '__main__':
    # region Args Parser

    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', type=str, required=True, default = '')
    parser.add_argument('--save_dir', type=str, required=True, default = '')
    parser.add_argument('--processed_dataset_dir', type=str, required=True, default = '')
    parser.add_argument('--rqvae_path', type=str, required=True, default = '')
    parser.add_argument('--model_path_0', type=str, required=True, default = '')
    parser.add_argument('--model_path_1', type=str, required=True, default = '')
    parser.add_argument('--model_path_2', type=str, required=True, default = '')
    parser.add_argument('--model_path_3', type=str, required=True, default = '')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--init_body_pose_code', type=int, default=40)
    parser.add_argument('--init_hands_pose_code', type=int, default=40)
    parser.add_argument('--local-rank', default=-1, type=int)

    args = parser.parse_args()

    # endregion

    # region Distribution Setting

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # endregion

    print("Preparing data...")
    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    new_save_dir = os.path.join(args.save_dir, audio_name)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(new_save_dir):
        os.mkdir(new_save_dir)
    shutil.copy(args.audio_path, os.path.join(new_save_dir, audio_name+'.wav'))

    data_processor = Processor(args=EasyDict(dict(
        data_dir=os.path.dirname(args.audio_path),
        save_dir=new_save_dir,
        fps=60,
        ds_rate=8
    )))
    data_processor.process_for_generation(audio_name=audio_name,
                                          audio_feature_scaler_path=os.path.join(args.processed_dataset_dir,
                                                                                 'audio_feature_scaler.sav'),
                                          motion_data_template_path=os.path.join(args.processed_dataset_dir,
                                                                                 'motion_data_template.pkl'))

    print("Generating...")
    generate(data_dir=new_save_dir, audio_name=audio_name, audio_path=args.audio_path, 
             gpt_path=[args.model_path_0, args.model_path_1, args.model_path_2, args.model_path_3], rqvae_path=args.rqvae_path, 
             use_gpu=args.use_gpu, init_body_pose_code=args.init_body_pose_code,  
             init_hands_pose_code = args.init_hands_pose_code, local_rank=args.local_rank)