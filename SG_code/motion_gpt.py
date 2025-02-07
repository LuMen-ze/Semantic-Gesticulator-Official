# region Import

import os
import Model
import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import itertools
import yaml
import random

import numpy as np

from tqdm import tqdm
from datetime import datetime
from Dataset.ma_seq import MASeq, Big_MASeq_Test_h5, Big_MASeq_Train_h5
from Utils.utils import load_train_data_BEAT, load_test_data_BEAT, visualize_and_write, load_train_info_BEAT_large, load_test_info_BEAT_large
from Utils.log import Logger

# endregion


def _loss_fn(x_target, x_pred):
    return torch.mean(torch.abs(x_pred - x_target)) 

class MoGPT:
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if self.config.cuda else 'cpu', config.local_rank)
        self._build()

    def train(self):
        self.model_vqvae.eval()
        self.model_gpt.train()
        if self.config.local_rank == 0:
            log = Logger(self.config, self.exp_dir)
        updates = 0

        print("we use vqvae-model:", self.config.vqvae_weight)
        self.model_vqvae.load_state_dict(torch.load(self.config.vqvae_weight)['model'], strict=False)
        if hasattr(self.config, 'init_weight') and (self.config.init_weight is not None) and (self.config.init_weight != ''):
            print('Use pretrained model')
            print(self.config.init_weight)
            self.model_gpt.load_state_dict(torch.load(self.config.init_weight)['model'], strict=False)

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

        for epoch in range(1, self.config.epochs+1):
            self.train_sampler.set_epoch(epoch)
            if self.config.local_rank == 0:
                log.set_progress(epoch, len(self.train_loader))

            for batch in self.train_loader:
                body_seqs, audio_feats = batch
                body_seqs = body_seqs.to(self.device)
                audio_feats = audio_feats.to(self.device)

                self.optimizer.zero_grad()

                if self.config.structure_gpt.name in ['CrossCondGPT2', 'CrossCondGPT2_2part']:
                    with torch.no_grad():
                        quants_pred = self.model_vqvae.module.encode(body_seqs)
                        if isinstance(quants_pred, tuple):
                            quants_input = tuple(
                                quants_pred[idx][0][:, :-1].clone().detach() for idx in range(len(quants_pred)))
                            quants_target = tuple(
                                quants_pred[idx][0][:, 1:].clone().detach() for idx in range(len(quants_pred)))
                        else:
                            quants_input = quants_pred[0][:, :-1].clone().detach()
                            quants_target = quants_pred[0][:, 1:].clone().detach()
                    
                    input_audio_cond = audio_feats[:, 1:, :]
                elif self.config.structure_gpt.name in ['Fine_GPT2', 'Fine_GPT2_2part']:
                    with torch.no_grad():
                        quants_pred = self.model_vqvae.module.encode(body_seqs)
                        if self.config.structure_gpt.layer_no == 1:
                            if isinstance(quants_pred, tuple):
                                quants_input = tuple(
                                    quants_pred[idx][0][:, :].clone().detach() for idx in range(len(quants_pred)))
                                quants_target = tuple(
                                    quants_pred[idx][1][:, :].clone().detach() for idx in range(len(quants_pred)))
                            else:
                                raise(ValueError)
                            
                        elif self.config.structure_gpt.layer_no == 2:
                            if isinstance(quants_pred, tuple):
                                quants_input_0 = tuple(
                                    quants_pred[idx][0][:, :].clone().detach() for idx in range(len(quants_pred)))
                                quants_input_1 = tuple(
                                    quants_pred[idx][1][:, :].clone().detach() for idx in range(len(quants_pred)))
                                quants_input = (quants_input_0, quants_input_1)
                                quants_target = tuple(
                                    quants_pred[idx][2][:, :].clone().detach() for idx in range(len(quants_pred)))
                            else:
                                raise(ValueError)
                        
                        elif self.config.structure_gpt.layer_no == 3:
                            if isinstance(quants_pred, tuple):
                                quants_input_0 = tuple(
                                    quants_pred[idx][0][:, :].clone().detach() for idx in range(len(quants_pred)))
                                quants_input_1 = tuple(
                                    quants_pred[idx][1][:, :].clone().detach() for idx in range(len(quants_pred)))
                                quants_input_2 = tuple(
                                    quants_pred[idx][2][:, :].clone().detach() for idx in range(len(quants_pred)))
                                quants_input = (quants_input_0, quants_input_1, quants_input_2)
                                quants_target = tuple(
                                    quants_pred[idx][3][:, :].clone().detach() for idx in range(len(quants_pred)))
                            else:
                                raise(ValueError)

                    input_audio_cond = audio_feats
                else:
                    raise ValueError

                logits, loss, metrics = self.model_gpt(quants_input, input_audio_cond, quants_target)
                
                cal_recon_loss = False
                if cal_recon_loss:
                    # [n, l, dim]
                    logits_up, logits_down, logits_hands = logits
                    code_up_one_hot = torch.nn.functional.gumbel_softmax(logits_up, tau=0.1, hard=True)
                    code_down_one_hot = torch.nn.functional.gumbel_softmax(logits_down, tau=0.1, hard=True)
                    code_hands_one_hot = torch.nn.functional.gumbel_softmax(logits_hands, tau=0.1, hard=True)

                    code_up_one_hot_first = F.one_hot(quants_input[0].long(), num_classes=512).float()
                    code_down_one_hot_first = F.one_hot(quants_input[1].long(), num_classes=512).float()
                    code_hands_one_hot_first = F.one_hot(quants_input[2].long(), num_classes=512).float()

                    code_up_one_hot = torch.cat((code_up_one_hot_first[:, 0:1], code_up_one_hot), dim=1)
                    code_down_one_hot = torch.cat((code_down_one_hot_first[:, 0:1], code_down_one_hot), dim=1)
                    code_hands_one_hot = torch.cat((code_hands_one_hot_first[:, 0:1], code_hands_one_hot), dim=1)

                    code_one_hot = (code_up_one_hot, code_down_one_hot, code_hands_one_hot)
                    predict_body = self.model_vqvae.module.one_hot_decode(code_one_hot, type='one_hot')
                    recon_loss = _loss_fn(body_seqs, predict_body)

                    velocity_loss = _loss_fn(body_seqs[:, 1:] - body_seqs[:, :-1], predict_body[:, 1:] - predict_body[:, :-1])
                    acceleration_loss = _loss_fn(body_seqs[:, 2:] + body_seqs[:, :-2] - 2 * body_seqs[:, 1:-1],
                                            predict_body[:, 2:] + predict_body[:, :-2] - 2 * predict_body[:, 1:-1])
                    
                    all_loss = loss + recon_loss + velocity_loss + acceleration_loss
                    all_loss.backward()
                    self.optimizer.step()

                    stats = {
                        'updates': updates,
                        'loss': all_loss.item(),
                        'loss_gpt': loss.item(),
                        'loss_recon': recon_loss.item(),
                        'loss_velocity': velocity_loss.item(),
                        'loss_acceleration': acceleration_loss.item(),
                        'accuracy_up': metrics['accuracy_up'].item(),
                        'accuracy_down': metrics['accuracy_down'].item(),
                        'accuracy_hands': metrics['accuracy_hands'].item()
                    }
                else:
                    loss.backward()
                    self.optimizer.step()

                    if 'accuracy_down' in metrics:
                        stats = {
                            'updates': updates,
                            'loss': loss.item(),
                            'accuracy_up': metrics['accuracy_up'].item(),
                            'accuracy_down': metrics['accuracy_down'].item(),
                            'accuracy_hands': metrics['accuracy_hands'].item()
                        }
                    else:
                        stats = {
                            'updates': updates,
                            'loss': loss.item(),
                            'accuracy_body': metrics['accuracy_body'].item(),
                            'accuracy_hands': metrics['accuracy_hands'].item()
                        }
                        
                if self.config.local_rank == 0:
                    log.update(stats)
                updates += 1

            checkpoint = {
                'model': self.model_gpt.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config,
                'epoch': epoch,
                'updates': updates
            }
            if ((epoch % self.config.save_per_epochs == 0) or (epoch == 1)) and (self.config.local_rank == 0):
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pt'))

            if (epoch % self.config.eval_per_epochs == 0) and (self.config.local_rank == 0) and not (hasattr(self.config, 'need_not_test_data') and self.config.need_not_test_data == 1):
                with torch.no_grad():
                    self.model_gpt.eval()
                    gts = []
                    results = []
                    quants = {}
                    for i_eval, batch_eval in enumerate(self.test_loader):
                        body_seqs_eval, audio_feats_eval = batch_eval
                        body_seqs_eval = body_seqs_eval.to(self.device)
                        audio_feats_eval = audio_feats_eval.to(self.device)

                        quants_gt = self.model_vqvae.module.encode(body_seqs_eval)
                        if isinstance(quants_gt, tuple):
                            init_z = tuple(quants_gt[i][0][:, :1] for i in range(len(quants_gt)))
                        else:
                            init_z = quants_gt[0][:, :1]

                        zs = self.model_gpt.module.sample(init_z, cond=audio_feats_eval[:, 1:, :],
                                               shift=self.config.sample_shift if hasattr(self.config, 'sample_shift') else None)

                        body_seqs_out = self.model_vqvae.module.decode(zs)

                        file_name = self.config.data.test_files[i_eval // self.config.data.samples_per_test] + '_' + str(i_eval % self.config.data.samples_per_test)
                        if isinstance(zs, tuple):
                            quants[file_name] = tuple(zs[idx][0].cpu().data.numpy()[0] for idx in range(len(zs)))
                        else:
                            quants[file_name] = zs[0].cpu().data.numpy()[0]

                        # For debug
                        z_gt_up = quants_gt[0][0].cpu().data.numpy()[0]
                        z_pred_up = quants[file_name][0].copy()
                        z_gt_down = quants_gt[1][0].cpu().data.numpy()[0]
                        z_pred_down = quants[file_name][1].copy()

                        gts.append(body_seqs_eval)
                        results.append(body_seqs_out)

                    visualize_and_write(results, gts, self.config, self.vis_dir, epoch, quants)

                self.model_gpt.train()

            self.schedular.step()

    def eval(self):
        self.model_vqvae.eval()
        self.model_gpt.eval()
        if self.config.local_rank == 0:
            log = Logger(self.config, self.exp_dir)
        updates = 0

        self.model_vqvae.load_state_dict(torch.load(self.config.vqvae_weight)['model'], strict=False)
        if hasattr(self.config, 'init_weight') and (self.config.init_weight is not None) and (self.config.init_weight != ''):
            print('Use pretrained model')
            print(self.config.init_weight)
            self.model_gpt.load_state_dict(torch.load(self.config.init_weight)['model'], strict=False)

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

        with torch.no_grad():
            self.model_gpt.eval()
            gts = []
            results = []
            quants = {}

            txt_path = os.path.join(self.vis_dir, "txt")
            npy_path = os.path.join(self.vis_dir, "npy")

            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
            if not os.path.exists(npy_path):
                    os.makedirs(npy_path)
            for i_eval, batch_eval in enumerate(self.test_loader):
                body_seqs_eval, audio_feats_eval = batch_eval
                body_seqs_eval = body_seqs_eval.to(self.device)
                audio_feats_eval = audio_feats_eval.to(self.device)

                quants_gt = self.model_vqvae.module.encode(body_seqs_eval)
                if isinstance(quants_gt, tuple):
                    init_z = tuple(quants_gt[i][0][:, :1] for i in range(len(quants_gt)))
                else:
                    init_z = quants_gt[0][:, :1]

                print("init_z is: ", init_z)
                
                zs = self.model_gpt.module.sample(init_z, cond=audio_feats_eval[:, 1:, :],
                                        shift=self.config.sample_shift if hasattr(self.config, 'sample_shift') else None)

                body_seqs_out = self.model_vqvae.module.decode(zs)

                file_name = self.config.data.test_files[i_eval // self.config.data.samples_per_test] + '_' + str(i_eval % self.config.data.samples_per_test)
                if isinstance(zs, tuple):
                    quants[file_name] = tuple(zs[idx][0].cpu().data.numpy()[0] for idx in range(len(zs)))
                else:
                    quants[file_name] = zs[0].cpu().data.numpy()[0]

                # For debug
                z_gt_up = quants_gt[0][0].cpu().data.numpy()[0]
                z_pred_up = quants[file_name][0].copy()
                z_gt_down = quants_gt[1][0].cpu().data.numpy()[0]
                z_pred_down = quants[file_name][1].copy()
                assert len(z_gt_up) == len(z_pred_up)

                code_index_path = os.path.join(txt_path, file_name)
                f = open(code_index_path, 'w')
                f.write("upper body:\n")
                f.write(str(z_pred_up))
                f.write("\nlower body:\n")
                f.write(str(z_pred_down))
                f.write("\nhands:\n")
                f.write(str(quants[file_name][2]))

                code_index_path_npy = os.path.join(npy_path, file_name)
                up_code = z_pred_up
                down_code = z_pred_down
                hand_code = quants[file_name][2]

                np.save(code_index_path_npy, np.concatenate((up_code, down_code, hand_code), axis=0).reshape(3,-1))

                gts.append(body_seqs_eval)
                results.append(body_seqs_out)

            epoch = 0
            need_gt_pos = 0
            visualize_and_write(results, gts, self.config, self.vis_dir, epoch, need_gt_pos, quants)


    def _build(self):
        self._dir_setting()
        self._build_model()
        if not (hasattr(self.config, 'need_not_train_data') and self.config.need_not_train_data):
            self._build_train_loader()
        if not (hasattr(self.config, 'need_not_test_data') and self.config.need_not_test_data):
            self._build_test_loader()
        self._build_optimizer()

    def _dir_setting(self):
        date_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.exp_dir = os.path.join('./Experiment', self.config.exp_name, self.config.log_name+'_'+date_time)
        os.makedirs(self.exp_dir, exist_ok=True)

        config_save_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_save_path, 'w') as yaml_file:
            yaml.dump(dict(self.config), yaml_file, default_flow_style=False)

        # os.makedirs(os.path.join('./Experiment', self.config.exp_name, 'Tensorboard_Log'))

        self.checkpoint_dir = os.path.join(self.exp_dir, "Checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.vis_dir = os.path.join(self.exp_dir, "Vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(os.path.join(self.vis_dir, "BVH"), exist_ok=True)

    def _build_model(self):
        print(f'Using {self.config.structure_vqvae.name} and {self.config.structure_gpt.name}')

        self.model_vqvae = torch.nn.parallel.DistributedDataParallel(
            getattr(Model, self.config.structure_vqvae.name)(self.config.structure_vqvae).cuda(),
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=True
        )
        self.model_gpt = torch.nn.parallel.DistributedDataParallel(
            getattr(Model, self.config.structure_gpt.name)(self.config.structure_gpt).cuda(),
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=True
        )

    def _build_train_loader(self):
        print("Build train loader")
        data = self.config.data
        if data.name == "BEAT" or data.name == "zeroeggs" or data.name == "mocap":
            print("Train with BEAT dataset")
            train_file_names, train_body_motions_len, train_audio_features_len = load_train_info_BEAT_large(
                data_dir=data.dir,
                test_files=data.test_files,
                window_size=data.seq_len_train,
                stride=data.stride_train
            )
        else:
            raise ValueError

        train_dataset = Big_MASeq_Train_h5(
            data_dir = data.dir,
            body_motions_name = train_file_names, 
            body_motions_len = train_body_motions_len, 
            window_size = data.seq_len_train, 
            stride = data.stride_train,
            ds_rate = data.ds_rate
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            # shuffle=True,
            num_workers=16,
            pin_memory=True,
            sampler=train_sampler
        )

        self.train_loader = data_loader
        self.train_sampler = train_sampler

    def _build_test_loader(self):
        print("Build test loader")
        data = self.config.data
        if data.name == "BEAT" or data.name == "zeroeggs" or data.name == "mocap":
            print("Test with BEAT dataset")
            file_names, test_body_motions_len, test_audio_features_len = load_test_info_BEAT_large(
                data_dir=data.dir,
                test_files=data.test_files,
                window_size=data.seq_len_test,
                stride=data.stride_test
            )
        else:
            raise ValueError

        test_dataset = Big_MASeq_Test_h5(
            data_dir = data.dir,
            body_motions_name = file_names, 
            body_motions_len = test_body_motions_len, 
            window_size = data.seq_len_test, 
            stride = data.stride_test,
            ds_rate = data.ds_rate,
            samples_per_test = data.samples_per_test
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False
        )

        self.test_loader = data_loader

    def _build_optimizer(self):
        config = self.config.optimizer

        self.optimizer = getattr(torch.optim, config.type)(itertools.chain(self.model_gpt.module.parameters()),
                                                           **config.kwargs)
        
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)
