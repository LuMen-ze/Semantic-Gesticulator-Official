# region Import

import os
import json
import pickle
import librosa
import argparse
import copy

from scipy.spatial.transform import Rotation as R 

import numpy as np

from sklearn.preprocessing import StandardScaler
from Utils.BVH_loader import load as bvh_load
from Utils.MathHelper import MathHelper
from Utils.audio_feature import AudioFeature
from Utils.utils import fit_and_standardize, standardize, save_standard_scalers, load_standard_scalers, sub_root_offset_BEAT, align_quaternion

# endregion

class Processor:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.fps = args.fps
        self.ds_rate = args.ds_rate

        os.makedirs(self.save_dir, exist_ok=True)

    def process4beat(self):
        files = self._scan_data_dir4beat()  # {file_name: {dir, speaker_id}}
        num_files = len(files)

        features_dir = os.path.join(self.save_dir, 'Features')
        os.makedirs(features_dir, exist_ok=True)
        body_motion_dir = os.path.join(features_dir, 'Body')
        os.makedirs(body_motion_dir, exist_ok=True)
        audio_feature_dir = os.path.join(features_dir, 'Audio')
        os.makedirs(audio_feature_dir, exist_ok=True)
        
        for i, (name, info) in enumerate(files.items()):
            print(f"Processing {name} ({i+1}/{num_files})")

            if i == 0:
                print(f"Select {name} as motion data template")
                data = bvh_load(os.path.join(info["dir"], name+".bvh"), ignore_root_offset=False)
                data = data.resample(self.fps)
                data._num_frames = 1
                data._joint_rotation = np.zeros((1, data.num_joints, 4))
                data._joint_translation = np.zeros((1, data.num_joints, 3))
                data.reset_global_info()
                with open(os.path.join(self.save_dir, "motion_data_template.pkl"), 'wb') as f:
                    pickle.dump(data, f)

            if os.path.exists(os.path.join(body_motion_dir, name+'.npy')) and i != 0:
                body_motion = np.load(os.path.join(body_motion_dir, name+'.npy'))
            else:
                body_motion, joint_names = self._extract_body_motion_features(os.path.join(info["dir"], name+".bvh"))
                print(body_motion)
                np.save(os.path.join(body_motion_dir, name+'.npy'), body_motion)

            if os.path.exists(os.path.join(audio_feature_dir, name + '.npy')) and i != 0:
                print("has audio")
                audio_feature = np.load(os.path.join(audio_feature_dir, name + '.npy'))
            else:
                print("no audio ")
                audio_feature, audio_feature_table = self._extract_audio_features(os.path.join(info["dir"], name+".wav"),
                                                                                  sr=(512*self.fps//self.ds_rate))
                np.save(os.path.join(audio_feature_dir, name + '.npy'), audio_feature)

    def process4zeroeggs(self, zeggs_path):
        files = self._scan_data_dir4zeroeggs(zeggs_path)  # {file_name: {dir, speaker_id}}
        num_files = len(files)

        features_dir = os.path.join(self.save_dir, 'Features')
        os.makedirs(features_dir, exist_ok=True)
        body_motion_dir = os.path.join(features_dir, 'Body')
        os.makedirs(body_motion_dir, exist_ok=True)
        audio_feature_dir = os.path.join(features_dir, 'Audio')
        os.makedirs(audio_feature_dir, exist_ok=True)

        body_motions, audio_features = [], []  # num_files, time, feature
        joint_names, audio_feature_table = [], []

        for i, (key, value) in enumerate(files.items()):
            print(f"Processing {key} ({i+1}/{num_files})")

            if i == 0:
                print(f"Select {key} as motion data template")
                data = bvh_load(os.path.join(value["dir"], key+".bvh"), ignore_root_offset=False)
                data = data.resample(self.fps)
                data._num_frames = 1
                data._joint_rotation = np.zeros((1, data.num_joints, 4))
                data._joint_translation = np.zeros((1, data.num_joints, 3))
                data.reset_global_info()
                with open(os.path.join(self.save_dir, "motion_data_template.pkl"), 'wb') as f:
                    pickle.dump(data, f)

            if os.path.exists(os.path.join(body_motion_dir, key+'.npy')) and i != 0:
                body_motion = np.load(os.path.join(body_motion_dir, key+'.npy'))
            else:
                body_motion, joint_names = self._extract_body_motion_features(os.path.join(value["dir"], key+".bvh"))
                np.save(os.path.join(body_motion_dir, key+'.npy'), body_motion)
            
            if os.path.exists(os.path.join(audio_feature_dir, key + '.npy')) and i != 0:
                print("has audio")
                audio_feature = np.load(os.path.join(audio_feature_dir, key + '.npy'))
            else:
                print("no audio")
                audio_feature, audio_feature_table = self._extract_audio_features(os.path.join(value["dir"], key+".wav"),
                                                                                  sr=(512*self.fps//self.ds_rate))
                np.save(os.path.join(audio_feature_dir, key + '.npy'), audio_feature)

    def process4mocap(self, sg_path):
        files = self._scan_data_dir4zeroeggs(sg_path) 
        num_files = len(files)
        print(num_files)

        features_dir = os.path.join(self.save_dir, 'Features')
        os.makedirs(features_dir, exist_ok=True)
        body_motion_dir = os.path.join(features_dir, 'Body')
        os.makedirs(body_motion_dir, exist_ok=True)

        body_motions = []  # num_files, time, feature
        joint_names = []
        
        for i, (key, value) in enumerate(files.items()):
            print(f"Processing {key} ({i+1}/{num_files})")

            if i == 0:
                print(f"Select {key} as motion data template")
                data = bvh_load(os.path.join(value["dir"], key+".bvh"), ignore_root_offset=False)
                data = data.resample(self.fps)
                data._num_frames = 1
                data._joint_rotation = np.zeros((1, data.num_joints, 4))
                data._joint_translation = np.zeros((1, data.num_joints, 3))
                data.reset_global_info()
                with open(os.path.join(self.save_dir, "motion_data_template.pkl"), 'wb') as f:
                    pickle.dump(data, f)

            if os.path.exists(os.path.join(body_motion_dir, key+'.npy')) and i != 0:
                body_motion = np.load(os.path.join(body_motion_dir, key+'.npy'))
            else:
                body_motion, joint_names = self._extract_body_motion_features(os.path.join(value["dir"], key+".bvh"))
                np.save(os.path.join(body_motion_dir, key+'.npy'), body_motion)

    def process_for_generation(self, audio_name, audio_feature_scaler_path, motion_data_template_path, speaker_id=1):
        audio_features, audio_feature_table = self._extract_audio_features(os.path.join(self.data_dir, audio_name+'.wav'),
                                                                           sr=(512*self.fps//self.ds_rate))
        audio_feature_scaler = load_standard_scalers(audio_feature_scaler_path)
        audio_features = standardize([audio_features], audio_feature_scaler)  # num_files(1), time, feature

        np.save(os.path.join(self.save_dir, "audio_features.npy"), np.array(audio_features))

        with open(motion_data_template_path, 'rb') as f:
            motion_data_template = pickle.load(f)
        num_joints = motion_data_template.num_joints

        config = dict(
            data_dir=self.data_dir,
            save_dir=self.save_dir,
            fps=self.fps,
            ds_rate=self.ds_rate,
            num_joints=num_joints,
            audio_feature_table=audio_feature_table,
            num_files=len(audio_features),
            duration_files=np.sum([audio_feature.shape[0] for audio_feature in audio_features]) / (self.fps / self.ds_rate),
            file_info={}
        )
        config['file_info'][audio_name] = dict(
            dir=self.data_dir,
            speaker_id=speaker_id,
            num_frames=audio_features[0].shape[0],
            index=0
        )
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def _scan_data_dir4beat(self, files_path):
        """ Scan Data Dir & Get File Name, Dir, and Speaker ID. """

        files = {}
        for r, d, f in os.walk(files_path):
            for n in f:
                if os.path.splitext(n)[1] == ".bvh":
                    files[os.path.splitext(n)[0]] = {"dir": r,
                                                     "speaker_id": int(os.path.basename(r))}
        return files

    def _scan_data_dir4zeroeggs(self, files_path):
        """ Scan Data Dir & Get File Name, Dir, and Speaker ID. """

        files = {}
        for r, d, f in os.walk(files_path):
            for n in f:
                if os.path.splitext(n)[1] == ".bvh":
                    files[os.path.splitext(n)[0]] = {"dir": r}
        return files
    
    def _extract_body_motion_features(self, bvh_path):
        data = bvh_load(bvh_path, ignore_root_offset=False)
        data = data.resample(self.fps)

        # data = data.to_facing_coordinate_for_first_frame()
        root_offset = copy.deepcopy(data.joint_translation[:, 0])  # num_frames, 3
        root_offset = root_offset[1:, :] - root_offset[:-1, :]
        zero_padding = np.zeros(3)
        root_offset = np.insert(root_offset, 0, zero_padding, axis = 0)

        data._joint_translation[:, 0, :] = 0
        data.recompute_joint_global_info()

        assert data.joint_rotation.shape[0] == data.num_frames

        num_frames = data.num_frames
        joint_names = data.joint_names  # J

        # flip all the quats to the same hemisphere
        rot = data.joint_rotation
        rot = MathHelper.flip_quat_arr_by_dot(rot)
        rot = np.array(rot[0]) # num_frames, J, 4
        data._joint_rotation = rot

        positions = data.joint_position  # num_frames, J, 3
        positions[:, 0] = root_offset
        expmaps = MathHelper.quat_to_expmap_sg(data._joint_rotation)  # num_frames, J, 3

        features = expmaps.reshape(num_frames, -1)  # num_frames, 6J
        features = np.concatenate([features, root_offset], axis=-1)

        return features, joint_names

    def _extract_audio_features(self, wav_path, sr):
        assert self.fps/self.ds_rate == (sr/512)

        data, _ = librosa.load(wav_path, sr=sr)

        melspe_db = AudioFeature.get_melspectrogram(data, sr)
        mfcc = AudioFeature.get_mfcc(melspe_db)  # 20, num_frames
        mfcc_delta = AudioFeature.get_mfcc_delta(mfcc)  # 20, num_frames

        audio_harmonic, audio_percussive = AudioFeature.get_hpss(data)
        chroma_cqt = AudioFeature.get_chroma_cqt(audio_harmonic, sr, octave=(7 if sr == 60*512 else 5))  # 12, num_frames. octave is 7 if sr=60*512 else 5

        onset_env = AudioFeature.get_onset_strength(audio_percussive, sr)  # num_frames,

        tempogram = AudioFeature.get_tempogram(onset_env, sr)  # 384, num_frames

        features = np.concatenate([
            mfcc,
            mfcc_delta,
            chroma_cqt,
            onset_env.reshape(1, -1),
            tempogram
        ], axis=0).transpose()  # num_frames, 437

        feature_table = {
            "mfcc": 20,
            "mfcc_delta": 20,
            "chroma_cqt": 12,
            "onset_env": 1,
            "tempogram": 384
        }

        return features, feature_table

    def _align(self, motion_feature_list, audio_feature):
        min_num_motion_frames = np.min([feature.shape[0] for feature in motion_feature_list] + [audio_feature.shape[0]*self.ds_rate])
        assert min_num_motion_frames > 0
        min_num_audio_frames = min_num_motion_frames // self.ds_rate
        min_num_motion_frames = min_num_audio_frames * self.ds_rate

        return [feature[:min_num_motion_frames, :] for feature in motion_feature_list] + [audio_feature[:min_num_audio_frames, :]]


class Processor_h5:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.fps = args.fps
        self.ds_rate = args.ds_rate

        os.makedirs(self.save_dir, exist_ok=True)

    def process(self, mode):
        self.mode = mode

        files = self._scan_data_dir4zeroeggs()  # {file_name: {dir, speaker_id}}
        num_files = len(files)

        features_dir = os.path.join(self.save_dir, 'Features')
        body_motion_dir = os.path.join(features_dir, 'Body')
        audio_feature_dir = os.path.join(features_dir, 'Audio')

        if self.mode != 'save_h5_every_part':
            body_motions, audio_features = [], [] # num_files, time, feature
            for i, (name, info) in enumerate(files.items()):
                print(f"Processing {name} ({i+1}/{num_files})")

                no_audio = 0
                if os.path.exists(os.path.join(body_motion_dir, name+'.npy')):
                    body_motion = np.load(os.path.join(body_motion_dir, name+'.npy'))
                else:
                    print("no such body info, name is ", name)
                if os.path.exists(os.path.join(audio_feature_dir, name + '.npy')):
                    audio_feature = np.load(os.path.join(audio_feature_dir, name + '.npy'))
                else:
                    print("no such audio info, name is ", name)
                    audio_feature = []
                    no_audio = 1

                if no_audio == 0:
                    aligned_feature_list = [feature for feature in self._align([body_motion], audio_feature)]
                    # body_motions.append(sub_root_offset_BEAT(aligned_feature_list[0]))
                    body_motions.append(aligned_feature_list[0])
                    audio_features.append(aligned_feature_list[1])
                else:
                    body_motions.append(body_motion)
                    audio_features.append(audio_feature)

            if self.mode == 'value':
                have_motion_data = 0
                flat_motions = np.concatenate(body_motions, axis=0)
                if len(flat_motions) > 0:
                    body_scaler = StandardScaler().fit(flat_motions)
                    have_motion_data = 1
                
                new_audio_features = []
                have_audio_data = 0
                for i in audio_features:
                    if not type(i) == list:
                        new_audio_features.append(i)
                if len(new_audio_features) > 0:
                    flat_audio = np.concatenate(new_audio_features, axis=0)
                    audio_feature_scaler = StandardScaler().fit(flat_audio)
                    have_audio_data = 1

                if have_audio_data == 1:
                    save_standard_scalers(scalers=audio_feature_scaler, save_paths=os.path.join(self.save_dir, "audio_feature_scaler.sav"))
                if have_motion_data == 1:
                    save_standard_scalers(scalers=body_scaler, save_paths=os.path.join(self.save_dir, "body_scaler.sav"))

            elif self.mode == 'num':
                print("body shape is: ", len(body_motions))
                all_frame_num = 0
                for i in range(len(body_motions)):
                    all_frame_num = all_frame_num + body_motions[i].shape[0]
                print(all_frame_num)

            elif self.mode == 'config':
                for i, (file_name, body_motion, audio_feature) in enumerate(zip(files.keys(), body_motions, audio_features)):
                    files[file_name]["num_motion_frames"] = body_motion.shape[0]
                    if type(audio_feature) == list:
                        files[file_name]["num_audio_frames"] = len(audio_feature)
                    else:
                        files[file_name]["num_audio_frames"] = audio_feature.shape[0]
                    files[file_name]["index"] = i  # {file_name: {dir, speaker_id, num_frames, index}}

                config = dict(
                    data_dir=self.data_dir,
                    save_dir=self.save_dir,
                    fps=self.fps,
                    ds_rate=self.ds_rate,
                    num_files=len(body_motions),
                    duration_files=np.sum([body_motion.shape[0] for body_motion in body_motions])/self.fps,
                    file_info=files
                )
                with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
            
            elif self.mode == 'save_npy_to_one':
                motion_scaler = load_standard_scalers(scaler_paths=os.path.join(self.save_dir, "body_scaler.sav"))
                body_motions = standardize(body_motions, motion_scaler)
                np.save(os.path.join(self.save_dir, "body_motions_1_15.npy"), body_motions, allow_pickle=True)

                audio_scaler = load_standard_scalers(scaler_paths=os.path.join(self.save_dir, "audio_feature_scaler.sav"))
                audio_features = standardize(audio_features, audio_scaler)
                np.save(os.path.join(self.save_dir, "audio_features_1_15.npy"), audio_features, allow_pickle=True)
        
        elif self.mode == 'save_h5_every_part':
            print("yeah")
            import h5py
            h5_dir = os.path.join(self.save_dir, 'h5')
            h5_dir_audio = os.path.join(self.save_dir, 'h5_audio')
            os.makedirs(h5_dir, exist_ok=True)
            os.makedirs(h5_dir_audio, exist_ok=True)

            motion_scaler = load_standard_scalers(scaler_paths=os.path.join(self.save_dir, "body_scaler.sav"))

            for i, (name, info) in enumerate(files.items()):
                print(f"Processing {name} ({i+1}/{num_files})")

                no_audio = 0
                if os.path.exists(os.path.join(body_motion_dir, name+'.npy')):
                    body_motion = np.load(os.path.join(body_motion_dir, name+'.npy'))
                else:
                    print("no such body info, name is ", name)
                if os.path.exists(os.path.join(audio_feature_dir, name + '.npy')):
                    audio_feature = np.load(os.path.join(audio_feature_dir, name + '.npy'))
                else:
                    print("no such audio info, name is ", name)
                    no_audio = 1
                    audio_feature = []

                if no_audio == 0:
                    audio_scaler = load_standard_scalers(scaler_paths=os.path.join(self.save_dir, "audio_feature_scaler.sav"))
                    
                    aligned_feature_list = [feature for feature in self._align([body_motion], audio_feature)]
                    # body_motions.append(sub_root_offset_BEAT(aligned_feature_list[0]))
                    sep_body_motion = aligned_feature_list[0]
                    sep_body_motion = motion_scaler.transform(sep_body_motion)
                    
                    sep_audio_feature = aligned_feature_list[1]
                    sep_audio_feature = audio_scaler.transform(sep_audio_feature)

                    file_name = name + '.h5'
                    file_path_motion = os.path.join(h5_dir, file_name) 
                    file_path_audio = os.path.join(h5_dir_audio, file_name) 

                    if not os.path.exists(file_path_motion):
                        with h5py.File(file_path_motion, 'w') as f:
                            f['value'] = sep_body_motion
                        f.close()

                    if not os.path.exists(file_path_audio):
                        with h5py.File(file_path_audio, 'w') as f:
                            f['value'] = sep_audio_feature
                        f.close()
                
                else:
                    sep_body_motion = motion_scaler.transform(body_motion)
                    file_name = name + '.h5'
                    file_path_motion = os.path.join(h5_dir, file_name) 
                    if not os.path.exists(file_path_motion):
                        with h5py.File(file_path_motion, 'w') as f:
                            f['value'] = sep_body_motion
                        f.close()

    def _scan_data_dir(self):
        """ Scan Data Dir & Get File Name, Dir, and Speaker ID. """

        files = {}
        for r, d, f in os.walk(self.data_dir):
            for n in f:
                if os.path.splitext(n)[1] == ".bvh":
                    files[os.path.splitext(n)[0]] = {"dir": r,
                                                     "speaker_id": int(os.path.basename(r))}

        return files

    def _scan_data_dir4zeroeggs(self):
        """ Scan Data Dir & Get File Name, Dir, and Speaker ID. """

        files = {}
        for r, d, f in os.walk(self.data_dir):
            # print(r,d,f)
            for n in f:
                if os.path.splitext(n)[1] == ".bvh":
                    files[os.path.splitext(n)[0]] = {"dir": r}
        return files
    
    def _align(self, motion_feature_list, audio_feature):
        min_num_motion_frames = np.min([feature.shape[0] for feature in motion_feature_list] + [audio_feature.shape[0]*self.ds_rate])
        assert min_num_motion_frames > 0
        min_num_audio_frames = min_num_motion_frames // self.ds_rate
        min_num_motion_frames = min_num_audio_frames * self.ds_rate

        return [feature[:min_num_motion_frames, :] for feature in motion_feature_list] + [audio_feature[:min_num_audio_frames, :]]


if __name__ == '__main__':
    # region Args Parser
    print("start")
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='Data/SG_Data/zeroeggs')
    parser.add_argument('--save_dir', type=str, default='Data/SG_processed')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--ds_rate', type=int, default=8)  # down sample rate of motion residual vq-vae encoder

    args = parser.parse_args()

    # endregion

    processor = Processor(args)
    sg_path = os.path.join(args.data_dir, 'SG')
    zeggs_path = os.path.join(args.data_dir, 'zeroeggs')
    processor.process4mocap(sg_path)
    processor.process4zeroeggs(zeggs_path)

    # process raw data to h5 files
    processor_h5 = Processor_h5(args)
    processor_h5.process(mode='value')
    processor_h5.process(mode='config')
    processor_h5.process(mode='save_h5_every_part')