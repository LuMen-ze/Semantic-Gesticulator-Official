# region Import

import numpy as np

from torch.utils.data import Dataset
import torch
import h5py
import os

# endregion


class MoSeq(Dataset):
    def __init__(self, body_motions, face_motions):
        self.body_motions = np.float32(body_motions)
        self.face_motions = np.float32(face_motions)
        assert self.body_motions.shape[: -1] == self.face_motions.shape[: -1]

    def __len__(self):
        return len(self.body_motions)

    def __getitem__(self, index):
        return {
            "body": self.body_motions[index],
            "face": self.face_motions[index]
        }


class Big_MoSeq_Test_h5(Dataset):
    def __init__(self, data_dir, body_motions_name, body_motions_len, window_size, stride, samples_per_test):
        self.data_dir = data_dir
        self.h5_dir = os.path.join(data_dir, 'h5')
        self.body_motions_name = body_motions_name
        self.body_motions_len = body_motions_len
        self.window_size = window_size
        self.stride = stride
        self.samples_per_test = samples_per_test

    def __len__(self):
        return len(self.body_motions_name) * self.samples_per_test

    def __getitem__(self, index):
        bvh_index = index // self.samples_per_test
        block_index = index % self.samples_per_test
        start_frame_num = self.stride * block_index

        file_name = self.body_motions_name[bvh_index] + '.h5'
        file_path = os.path.join(self.h5_dir, file_name)
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                motion_block = f['value'][start_frame_num: start_frame_num+self.window_size]
        else:
            raise(NotImplementedError)

        print("this block is:", bvh_index, block_index, self.window_size, np.array(motion_block).shape)

        return np.array(motion_block, dtype = np.float32)
    

# if data length is smaller than window_size, then it will be ignored
class Big_MoSeq_Train_h5(Dataset):
    def __init__(self, data_dir, body_motions_name, body_motions_len, window_size, stride):
        # self.body_motions = np.float32(body_motions)
        self.data_dir = data_dir
        self.h5_dir = os.path.join(data_dir, 'h5')
        self.body_motions_name = body_motions_name
        self.body_motions_len = body_motions_len
        self.window_size = window_size
        self.stride = stride
        self.block_index2motion = [] # record which motion file each block belongs to
        self.start_block_index = np.zeros(len(self.body_motions_name), dtype = int) # record the first block number of each motion file
        block_length = 0
        for i in range(len(self.body_motions_name)):
            self.start_block_index[i] = block_length
            if self.body_motions_len[i] < self.window_size:
                continue
            length = (self.body_motions_len[i] - self.window_size) // self.stride + 1
            block_length += length
            for j in range(length):
                self.block_index2motion.append(i)

        self.length = block_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        bvh_index = self.block_index2motion[index]
        block_index = index - self.start_block_index[bvh_index]
        start_frame_num = self.stride * block_index

        file_name = self.body_motions_name[bvh_index] + '.h5'
        file_path = os.path.join(self.h5_dir, file_name)
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                motion_block = f['value'][start_frame_num: start_frame_num+self.window_size]
        else:
            print(file_path)
            raise(NotImplementedError)

        return np.array(motion_block, dtype = np.float32)