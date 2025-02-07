# region Import

import os
import csv
import h5py
import Model
import torch
import torch.nn
import torch.utils.data
import itertools
import random
import yaml
import numpy as np

from tqdm import tqdm
from datetime import datetime
from Dataset.motion_seq import Big_MoSeq_Train_h5, Big_MoSeq_Test_h5
from Utils.utils import load_train_motion_info_BEAT_large, load_test_motion_info_BEAT_large, visualize_and_write, visualize_and_write_for_single, visualize_and_write_no_gt
from Utils.log import Logger

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from Utils.ddp import is_main_process
from Utils.ema import EMA, CPUEMA


# endregion
def _loss_fn(x_target, x_pred):
    return torch.mean(torch.abs(x_pred - x_target))


class MoVQ:
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True

        dist_url = "env://" # default

        # only works with torch.distributed.launch // torch.run
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        # self.local_rank = local_rank

        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=self.world_size,
                rank=self.rank)
        
        # this will make all .cuda() calls work properly
        # torch.cuda.set_device(self.local_rank)
        # synchronizes all the threads to reach this point before moving on
        dist.barrier() 

        self.device = torch.device('cuda', self.local_rank if self.config.cuda else 'cpu')
        print(self.device, self.rank, self.world_size, self.local_rank)
        self._build()

    def train(self):
        self.model.train()
        log = Logger(self.config, self.exp_dir)
        updates = 0

        if hasattr(self.config, 'init_weight') and (self.config.init_weight is not None) and (self.config.init_weight != ''):
            print('Use pretrained model & optimizer')
            print(self.config.init_weight)
            self.model.load_state_dict(torch.load(self.config.init_weight)['model'], strict=False)

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

        for epoch in range(1, self.config.epochs+1):
            log.set_progress(epoch, len(self.train_loader))

            for batch in self.train_loader:
                body_seqs = batch.to(self.device)
                    
                self.optimizer.zero_grad()

                if self.config.structure.name in ['SepVQVAE_DDP_body_hands']:
                    _, loss, loss_body, loss_hands, metrics = self.model(body_seqs)
                    loss.backward()                 
                else:
                    _, loss, loss_up, loss_down, loss_hands, metrics = self.model(body_seqs)
                    loss.backward()

                self.optimizer.step()
                
                if self.config.structure.name in ['SepVQVAE_DDP_body_hands']:
                    stats = {
                        'updates': updates,
                        'loss': loss.item(),

                        'recons_loss_body': metrics[0]['recons_loss'].item(),
                        'recons_loss_hands': metrics[1]['recons_loss'].item(),
                        'commit_loss_body': metrics[0]['commit_loss'].item(),
                        'commit_loss_hands': metrics[1]['commit_loss'].item(),
                        'sep_commit_loss_mean_body': metrics[0]['sep_commit_loss_mean'].item(),
                        'sep_commit_loss_mean_hands': metrics[1]['sep_commit_loss_mean'].item(),
                        
                        'used_curr_body': metrics[0]['used_curr'].item(),
                        'used_curr_hands': metrics[1]['used_curr'].item(),
                        'usage_body': metrics[0]['usage'].item(),
                        'usage_hands': metrics[1]['usage'].item(),
                    }
                else:
                    raise NotImplementedError
                
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config,
                'epoch': epoch
            }
            if (epoch % self.config.save_per_epochs == 0) or (epoch == 1):
                if is_main_process():
                    torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pt'))
            if epoch % self.config.eval_per_epochs == 0 and not (hasattr(self.config, 'need_not_test_data') and self.config.need_not_test_data == 1):
                with torch.no_grad():
                    self.model.eval()
                    gts = []
                    results = []
                    quants = {}
                    for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc="Generating Test Gestures")):
                        body_seqs_eval = batch_eval.to(self.device)
                        
                        body_seq_out, _, _ = self.model(body_seqs_eval)

                        if self.config.global_vel:
                            raise NotImplementedError
                        gts.append(body_seqs_eval)
                        results.append(body_seq_out)

                        if self.config.structure.use_bottleneck:
                            file_name = self.config.data.test_files[i_eval//self.config.data.samples_per_test] + '_' + str(i_eval%self.config.data.samples_per_test)
                            quants_pred = self.model.module.encode(body_seqs_eval)
                            if isinstance(quants_pred, tuple):
                                quants[file_name] = tuple(quants_pred[idx][0].cpu().data.numpy()[0] for idx in range(len(quants_pred)))
                            else:
                                quants[file_name] = quants_pred[0].cpu().data.numpy()[0]
                        else:
                            quants = None
                        
                    visualize_and_write(results, gts, self.config, self.vis_dir, epoch, 0, quants)

                self.model.train()

            self.schedular.step()

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            if hasattr(self.config, 'init_weight') and (self.config.init_weight is not None) and (self.config.init_weight != ''):
                print('Use pretrained model')
                print(self.config.init_weight)
                self.model.load_state_dict(torch.load(self.config.init_weight)['model'], strict=False)      

            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if self.config.cuda:
                torch.cuda.manual_seed(self.config.seed)

            gts = []
            results = []
            quants = {}
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc="Generating Test Gestures")):
                body_seqs_eval = batch_eval.to(self.device)

                print(body_seqs_eval.shape)

                # body_seq_out, _, _ = self.model(body_seqs_eval)
                length_eval = (batch_eval.shape[1]//8) * 8
                # length_eval = min(length_eval, 240)
                body_seqs_eval = batch_eval[:, :length_eval]
                body_seqs_eval = body_seqs_eval.to(self.device)
                body_seq_out = self.model.module.inference(body_seqs_eval)

                if self.config.global_vel:
                    raise NotImplementedError
                
                gts.append(body_seqs_eval)
                results.append(body_seq_out)

                if self.config.structure.use_bottleneck:
                    file_name = self.config.data.test_files[i_eval//self.config.data.samples_per_test] + '_' + str(i_eval%self.config.data.samples_per_test)
                    quants_pred = self.model.module.encode(body_seqs_eval)
                    if isinstance(quants_pred, tuple):
                        quants[file_name] = tuple(quants_pred[idx][0].cpu().data.numpy()[0] for idx in range(len(quants_pred)))
                    else:
                        quants[file_name] = quants_pred[0].cpu().data.numpy()[0]
                else:
                    quants = None

            epoch = 0

            visualize_and_write(results, gts, self.config, self.vis_dir, epoch, 0, quants)

    def _build(self):
        self._dir_setting()
        self._build_model()
        # self._build_ema()

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
            yaml.dump(self.config, yaml_file, default_flow_style=False)

        # os.makedirs(os.path.join('./Experiment', self.config.exp_name, 'Tensorboard_Log'))

        self.checkpoint_dir = os.path.join(self.exp_dir, "Checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.vis_dir = os.path.join(self.exp_dir, "Vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(os.path.join(self.vis_dir, "BVH"), exist_ok=True)

    def _build_model(self):
        print(f'Using {self.config.structure.name}')
        self.model = getattr(Model, self.config.structure.name)(self.config.structure).to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if self.config.structure.name == 'SepVQVAER' or self.config.structure.name == 'SepVQVAER_2part' or self.config.structure.name == 'Transformer' or self.config.structure.name == 'Transformer_rq':
            find_unused_parameters = True
        else:
            find_unused_parameters = False
        print("build model", self.local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                      output_device=self.local_rank, find_unused_parameters=find_unused_parameters)
        print("build model done")

    def _build_ema(self):
        self.ema = None
        mu = self.config.ema.mu
        if self.config.ema.name == 'EMA':
            print("Using EMA")
            self.ema = EMA(self.model.parameters(), mu=mu)
        elif self.config.ema.name == 'CPUEMA' and dist.get_rank() == 0:
            print("Using CPU EMA")
            self.ema = CPUEMA(self.model.parameters(), mu=mu, freq=1)

    def _build_train_loader(self):
        data = self.config.data
        if data.name == "BEAT" or data.name == "zeroeggs" or data.name == "mocap":
            print("Train with BEAT dataset")
            train_body_motions_name, train_body_motions_len = load_train_motion_info_BEAT_large(
                data_dir=data.dir,
                test_files=data.test_files,
                window_size=data.seq_len_train,
                stride=data.stride_train
            )
        else:
            raise ValueError    

        motion_dataset = Big_MoSeq_Train_h5(
                data_dir = data.dir,
                body_motions_name = train_body_motions_name, 
                body_motions_len = train_body_motions_len, 
                window_size = data.seq_len_train, 
                stride = data.stride_train
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=motion_dataset,
            batch_size=self.config.batch_size,
            sampler=DistributedSampler(dataset=motion_dataset, shuffle=True),
            num_workers=8,
            pin_memory=True
        )

        self.train_loader = data_loader

    def _build_test_loader(self):
        data = self.config.data
        if data.name == "BEAT" or data.name == "zeroeggs" or data.name == "mocap":
            print("Test with BEAT dataset")
            test_body_motions_name, test_body_motions_len = load_test_motion_info_BEAT_large(
                data_dir=data.dir,
                test_files=data.test_files,
                window_size=data.seq_len_test,
                stride=data.stride_test
            )
        else:
            raise ValueError
        print(test_body_motions_name)
        motion_dataset = Big_MoSeq_Test_h5(
                data_dir = data.dir,
                body_motions_name = test_body_motions_name, 
                body_motions_len = test_body_motions_len, 
                window_size = data.seq_len_test, 
                stride = data.stride_test, 
                samples_per_test = data.samples_per_test
            )
        data_loader = torch.utils.data.DataLoader(
            dataset=motion_dataset,
            batch_size=1,
            sampler=DistributedSampler(dataset=motion_dataset, shuffle=False),
            shuffle=False
        )

        self.test_loader = data_loader

    def _build_optimizer(self):
        config = self.config.optimizer

        self.optimizer = getattr(torch.optim, config.type)(itertools.chain(self.model.module.parameters()),
                                                           **config.kwargs)
        self.optimizer.param_groups[0]['lr'] *= self.world_size ** 0.5
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

        if self.config.structure.name in ['SepVQVAER', 'SepVQVAER_2part']:
            print("This is SepVQVAER, so we freeze the encoder and decoder")
            for name, param in self.model.named_parameters():
                if 'root' not in name:
                    param.requires_grad = False
                else:
                    print(name, param.requires_grad)