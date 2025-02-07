# region Import

import numpy as np
import os
import h5py

from torch.utils.data import Dataset

# endregion


class MASeq(Dataset):
    def __init__(self, body_motions, audio_features):
        self.body_motions = body_motions
        self.audio_features = audio_features
        assert self.body_motions.shape[0] == self.audio_features.shape[0]

    def __len__(self):
        return len(self.body_motions)

    def __getitem__(self, index):
        return {
            "body": self.body_motions[index],
            "audio": self.audio_features[index]
        }


class Big_MASeq_Test_h5(Dataset):
    def __init__(self, data_dir, body_motions_name, body_motions_len, window_size, stride, ds_rate, samples_per_test):
        self.data_dir = data_dir
        self.motion_h5_dir = os.path.join(data_dir, 'h5')
        self.audio_h5_dir = os.path.join(data_dir, 'h5_audio')

        self.body_motions_name = body_motions_name
        self.body_motions_len = body_motions_len

        self.window_size = window_size
        self.stride = stride
        
        self.window_size_audio = window_size // ds_rate
        self.stride_audio = stride // ds_rate

        self.samples_per_test = samples_per_test

    def __len__(self):
        return len(self.body_motions_name) * self.samples_per_test

    def __getitem__(self, index):
        bvh_index = index // self.samples_per_test
        block_index = index % self.samples_per_test
        motion_start_frame_num = self.stride * block_index
        audio_start_frame_num = self.stride_audio * block_index
        # motion_block = self.body_motions[bvh_index][start_frame_num: start_frame_num+self.window_size]

        file_name = self.body_motions_name[bvh_index] + '.h5'

        motion_file_path = os.path.join(self.motion_h5_dir, file_name)
        if os.path.exists(motion_file_path):
            with h5py.File(motion_file_path, 'r') as f:
                motion_block = f['value'][motion_start_frame_num: motion_start_frame_num+self.window_size]
        else:
            print(motion_file_path)
            raise(NotImplementedError)
        
        audio_file_path = os.path.join(self.audio_h5_dir, file_name)
        if os.path.exists(audio_file_path):
            with h5py.File(audio_file_path, 'r') as f:
                audio_block = f['value'][audio_start_frame_num: audio_start_frame_num+self.window_size_audio]
        else:
            print(audio_file_path)
            raise(NotImplementedError)

        print("this block is:", bvh_index, block_index, self.window_size, np.array(motion_block).shape)

        return np.array(motion_block, dtype = np.float32), np.array(audio_block, dtype = np.float32)
    

# 对于长度小于window_size的数据会被丢掉
class Big_MASeq_Train_h5(Dataset):
    def __init__(self, data_dir, body_motions_name, body_motions_len, window_size, stride, ds_rate):
        # self.body_motions = np.float32(body_motions)
        self.data_dir = data_dir
        self.motion_h5_dir = os.path.join(data_dir, 'h5')
        self.audio_h5_dir = os.path.join(data_dir, 'h5_audio')

        self.body_motions_name = body_motions_name
        self.body_motions_len = body_motions_len

        self.window_size = window_size
        self.stride = stride

        self.window_size_audio = window_size // ds_rate
        self.stride_audio = stride // ds_rate

        self.block_index2motion = [] # 记录滑窗后的每个块归属于哪个动作文件
        self.start_block_index = np.zeros(len(self.body_motions_name), dtype = int) # 记录每个动作文件对应的滑窗后的首块号
        block_length = 0
        for i in range(len(self.body_motions_name)):
            self.start_block_index[i] = block_length
            if self.body_motions_len[i] < self.window_size:
                continue
            length = (self.body_motions_len[i] - self.window_size) // self.stride + 1
            block_length += length
            for j in range(length):
                self.block_index2motion.append(i)

        self.length = block_length # 滑窗后总块数

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        bvh_index = self.block_index2motion[index]
        block_index = index - self.start_block_index[bvh_index]
        motion_start_frame_num = self.stride * block_index
        audio_start_frame_num = self.stride_audio * block_index

        file_name = self.body_motions_name[bvh_index] + '.h5'
        motion_file_path = os.path.join(self.motion_h5_dir, file_name)
        if os.path.exists(motion_file_path):
            with h5py.File(motion_file_path, 'r') as f:
                motion_block = f['value'][motion_start_frame_num: motion_start_frame_num+self.window_size]
        else:
            print(motion_file_path)
            raise(NotImplementedError)
        
        audio_file_path = os.path.join(self.audio_h5_dir, file_name)
        if os.path.exists(audio_file_path):
            with h5py.File(audio_file_path, 'r') as f:
                audio_block = f['value'][audio_start_frame_num: audio_start_frame_num+self.window_size_audio]
        else:
            print(audio_file_path)
            raise(NotImplementedError)

        return np.array(motion_block, dtype = np.float32), np.array(audio_block, dtype = np.float32)