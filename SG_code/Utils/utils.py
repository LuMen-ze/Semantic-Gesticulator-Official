import os
import json

import numpy as np
import joblib as jl

from easydict import EasyDict
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
from .MathHelper import MathHelper


def quat_product(p:np.ndarray, q:np.ndarray):
    if p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError('operands should be quaternions')

    if len(p.shape) != len(q.shape):
        if len(p.shape) == 1:
            p.reshape([1]*(len(q.shape)-1)+[4])
        elif len(q.shape) == 1:
            q.reshape([1]*(len(p.shape)-1)+[4])
        else:
            raise ValueError('mismatching dimensions')

    is_flat = len(p.shape) == 1
    if is_flat:
        p = p.reshape(1,4)
        q = q.reshape(1,4)
    
    product = np.empty([ max(p.shape[i], q.shape[i]) for i in range(len(p.shape)-1) ] + [4], dtype=np.result_type(p.dtype, q.dtype))
    product[..., 3] = p[..., 3] * q[..., 3] - np.sum(p[..., :3] * q[..., :3], axis=-1)
    product[..., :3] = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
                      np.cross(p[..., :3], q[..., :3]))

    if is_flat:
        product = product.reshape(4)

    return product


def flip_vector(vt:np.ndarray, normal:np.ndarray, inplace:bool):
    vt = np.asarray(vt).reshape(-1,3)
    normal = np.asarray(normal).reshape(-1,3)
    if inplace:
        vt -= (2 * np.sum(vt*normal, axis=-1, keepdims=True)) * normal
        return vt
    else:
        return vt - (2 * np.sum(vt*normal, axis=-1, keepdims=True)) * normal


def flip_quaternion(qt:np.ndarray, normal:np.ndarray, inplace:bool):
    qt = np.asarray(qt).reshape(-1,4)
    normal = np.asarray(normal).reshape(-1,3)

    if not inplace:
        qt = qt.copy()
    flip_vector(qt[:,:3], normal, True)
    qt[:,-1] = -qt[:,-1]
    return qt


def align_angles(a:np.ndarray, degrees:bool, inplace:bool):
    ''' make the angles in the array continuous

        we assume the first dim of a is the time
    '''
    step = 360. if degrees else np.pi*2

    a = np.asarray(a)
    diff = np.diff(a, axis=0)
    num_steps = np.round(diff / step)
    num_steps = np.cumsum(num_steps, axis=0)
    if not inplace:
        a = a.copy()
    a[1:] -= num_steps * step

    return a


def align_quaternion(qt:np.ndarray, inplace:bool):
    ''' make q_n and q_n+1 in the same semisphere

        the first axis of qt should be the time
    '''
    qt = np.asarray(qt)
    if qt.shape[-1] != 4:
        raise ValueError('qt has to be an array of quaterions')
    
    if not inplace:
        qt = qt.copy()

    if qt.size == 4: # do nothing since there is only one quation
        return qt

    sign = np.sum(qt[:-1]*qt[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0,)

    qt[1:][sign < 0] *= -1

    return qt


def extract_heading_Y_up(q:np.ndarray):
    ''' extract the rotation around Y axis from given quaternions
        
        note the quaterions should be {(x,y,z,w)}
    '''
    q = np.asarray(q)
    shape = q.shape
    q = q.reshape(-1, 4)

    v = R(q,True,False).as_dcm()[:,:,1]

    #axis=np.cross(v,(0,1,0))
    axis = v[:,(2,1,0)]
    axis *= [-1,0,1]
    
    norms = np.linalg.norm(axis,axis=-1)
    scales = np.empty_like(norms)
    small_angle = (norms <= 1e-3)
    large_angle = ~small_angle

    scales[small_angle] = norms[small_angle] + norms[small_angle]**3 / 6
    scales[large_angle] = np.arccos(v[large_angle,1]) / norms[large_angle]

    correct = R.from_rotvec(axis*scales[:,None])

    heading = (correct*R(q,True,False)).as_quat()
    heading[heading[:,-1] < 0] *= -1

    return heading.reshape(shape)


def extract_heading_frame_Y_up(root_pos, root_rots):
    heading = extract_heading_Y_up(root_rots)

    pos = np.copy(root_pos)
    pos[...,1] = 0

    return pos,heading


def get_joint_color(names, left='r', right='b', otherwise='y'):
    matches = (
        ('l', 'r'),
        ('L', 'R'),
        ('left', 'right'),
        ('Left', 'Right'),
        ('LEFT', 'RIGHT')
    )

    def check(n, i):
        for m in matches:
            if n[:len(m[i])] == m[i] and m[1-i] + n[len(m[i]):] in names:
                return True
                
            if n[-len(m[i]):] == m[i] and n[:-len(m[i])] + m[1-i] in names:
                return True

        return False

    color = [left if check(n, 0) else right if check(n, 1) else otherwise for n in names]
    return color


def animate_motion_data(data, show_skeleton=True, show_animation=True, interval=1):
    if (not show_skeleton) and (not show_animation):
        return

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D


    parent_idx = np.array(data._skeleton_joint_parents)
    parent_idx[0] = 0

    joint_colors=get_joint_color(data.joint_names)
    
    if data.end_sites is not None:
        for i in range(len(joint_colors)):
            if i in data.end_sites:
                joint_colors[i] = 'k'

    #############################
    # draw skeleton
    if show_skeleton:
        ref_joint_positions = data.get_reference_pose()
        tmp = ref_joint_positions.reshape(-1,3)
        bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
        bound[1,:] -= bound[0,:]
        bound[1,:] = np.max(bound[1,:])
        bound[1,:] += bound[0,:]

        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot('111', projection='3d')
        
        #ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        pos = ref_joint_positions
        strokes = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]

        ax.auto_scale_xyz(bound[:,0], -bound[:,2], bound[:,1])

    ########################################
    # animate motion
    if show_animation:
        joint_pos = data._joint_position
        tmp = joint_pos[:1].reshape(-1,3)
        bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
        bound[1,:] -= bound[0,:]
        bound[1,:] = np.max(bound[1,:])
        bound[1,:] += bound[0,:]

        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot('111', projection='3d')
        
        #ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        pos = joint_pos[0]
        strokes = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]

        ax.auto_scale_xyz(bound[:,0], -bound[:,2], bound[:,1])

        def update_lines(num):
            for (i,p) in enumerate(parent_idx):
                strokes[i][0].set_data(joint_pos[num][(i,p),0], -joint_pos[num][(i,p),2])
                strokes[i][0].set_3d_properties(joint_pos[num][(i,p),1])
            plt.title('frame {num}'.format(num=num))

        line_ani = animation.FuncAnimation(
            fig, update_lines, joint_pos.shape[0],
            interval=interval, blit=False)
        line_ani.save('2.gif', fps = 60)

    plt.show()


def fit_and_standardize(data):
    seq_lens = np.array([seq.shape[0] for seq in data])
    flat = np.concatenate(data, axis=0)

    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat)

    data_standardized = np.split(scaled, seq_lens.cumsum()[:-1])

    return data_standardized, scaler


def standardize(data, scaler):
    seq_lens = np.array([seq.shape[0] for seq in data])
    flat = np.concatenate(data, axis=0)

    scaled = scaler.transform(flat)

    data_standardized = np.split(scaled, seq_lens.cumsum()[:-1])

    return data_standardized


def inv_standardize(data, scaler):
    seq_lens = np.array([seq.shape[0] for seq in data])
    flat = np.concatenate(data, axis=0)

    scaled = scaler.inverse_transform(flat)

    data_inv_standardized = np.split(scaled, seq_lens.cumsum()[:-1])

    return data_inv_standardized


def load_standard_scalers(scaler_paths="./scaler.sav"):
    if isinstance(scaler_paths, str):
        return jl.load(scaler_paths)
    elif isinstance(scaler_paths, list):
        return [jl.load(p) for p in scaler_paths]
    else:
        raise TypeError


def save_standard_scalers(scalers, save_paths="./scaler.sav"):
    if isinstance(save_paths, str):
        jl.dump(scalers, save_paths)
    elif isinstance(save_paths, list):
        for s, p in zip(scalers, save_paths):
            jl.dump(s, p)
    else:
        raise TypeError


def sub_root_offset_BEAT(body_motion):
    t = body_motion.shape[0]
    body_motion = body_motion.reshape(t, -1, 6)  # time, J, 6
    j = body_motion.shape[1]
    root_pos = body_motion[:, 0, :3].copy()  # time, 3
    body_motion[:, :, :3] = body_motion[:, :, :3] - np.repeat(root_pos, j, axis=0).reshape((t, j, 3))
    body_motion[:, 0, :3] = root_pos
    body_motion = body_motion.reshape(t, -1)  # time, 6J

    return body_motion


def add_root_offset_BEAT(body_motion):
    t = body_motion.shape[0]
    body_motion = body_motion.reshape(t, -1, 6)  # time, J, 6
    j = body_motion.shape[1]
    root_pos = body_motion[:, 0, :3].copy()  # time, 3
    body_motion[:, :, :3] = body_motion[:, :, :3] + np.repeat(root_pos, j, axis=0).reshape((t, j, 3))
    body_motion[:, 0, :3] = root_pos
    body_motion = body_motion.reshape(t, -1)  # time, 6J

    return body_motion


# output: [block, frames, features]
def load_train_data_BEAT(data_dir, test_files, window_size, stride):
    body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    face_motions = np.load(os.path.join(data_dir, "face_motions.npy"), allow_pickle=True)
    audio_features = np.load(os.path.join(data_dir, "audio_features.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    assert stride % ds_rate == 0
    window_size_audio = window_size // ds_rate
    stride_audio = stride // ds_rate

    body_motion_seqs, face_motion_seqs, audio_feature_seqs = [], [], []  # num_seqs, window_size, feature
    
    train_files = list(set(config.file_info.keys()) - set(test_files))
    # train_files = list(set(config.file_info.keys()))
    
    for name in train_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames
        audio_len = config.file_info[name].num_audio_frames
        counter_m = 0
        counter_a = 0
        for s in range(0, motion_len-window_size+1, stride):
            body_motion_seqs.append(body_motions[index][s: s + window_size, :])
            face_motion_seqs.append(face_motions[index][s: s + window_size, :])
            counter_m += 1
        for sa in range(0, audio_len-window_size_audio+1, stride_audio):
            audio_feature_seqs.append(audio_features[index][sa: sa + window_size_audio, :])
            counter_a += 1
        assert counter_m == counter_a

    return np.array(body_motion_seqs, dtype=np.float32), \
           np.array(face_motion_seqs, dtype=np.float32), \
           np.array(audio_feature_seqs, dtype=np.float32)


def load_test_data_BEAT(data_dir, test_files, window_size, stride, samples_per_test):
    body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    face_motions = np.load(os.path.join(data_dir, "face_motions.npy"), allow_pickle=True)
    audio_features = np.load(os.path.join(data_dir, "audio_features.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    assert stride % ds_rate == 0
    window_size_audio = window_size // ds_rate
    stride_audio = stride // ds_rate

    body_motion_seqs, face_motion_seqs, audio_feature_seqs = [], [], []  # num_seqs, window_size, feature
    for name in test_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames
        audio_len = config.file_info[name].num_audio_frames
        counter_m = 0
        counter_a = 0
        for s in range(0, motion_len-window_size+1, stride):
            body_motion_seqs.append(body_motions[index][s: s+window_size, :])
            face_motion_seqs.append(face_motions[index][s: s + window_size, :])
            counter_m += 1
            if counter_m >= samples_per_test:
                break
        for sa in range(0, audio_len-window_size_audio+1, stride_audio):
            audio_feature_seqs.append(audio_features[index][sa: sa + window_size_audio, :])
            counter_a += 1
            if counter_a >= samples_per_test:
                break
        assert counter_m == counter_a

    return np.array(body_motion_seqs, dtype=np.float32), \
           np.array(face_motion_seqs, dtype=np.float32), \
           np.array(audio_feature_seqs, dtype=np.float32)


def load_train_motion_data_BEAT_large(data_dir, test_files, window_size, stride):
    body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    assert stride % ds_rate == 0

    body_motion_seqs = [] 
    body_motion_len = []
    
    train_files = list(set(config.file_info.keys()) - set(test_files))
    # train_files = list(set(config.file_info.keys()))
    for name in train_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames

        body_motion_seqs.append(body_motions[index])
        body_motion_len.append(motion_len)

    return body_motion_seqs, np.array(body_motion_len, dtype=int)


def load_test_motion_data_BEAT_large(data_dir, test_files, window_size, stride):
    body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    assert stride == window_size

    body_motion_seqs = [] 
    body_motion_len = []
        
    for name in test_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames

        body_motion_seqs.append(body_motions[index])
        body_motion_len.append(motion_len)

    return body_motion_seqs, np.array(body_motion_len, dtype=int)


def load_both_train_test_motion_data_BEAT_large(data_dir, test_files, window_size_train, stride_train, window_size_test, stride_test):
    body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size_train % ds_rate == 0
    assert stride_train % ds_rate == 0
    assert window_size_test % ds_rate == 0
    assert stride_test == window_size_test

    train_body_motion_seqs = [] 
    train_body_motion_len = []
    
    train_files = list(set(config.file_info.keys()) - set(test_files))
    # train_files = list(set(config.file_info.keys()))
    for name in train_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames

        train_body_motion_seqs.append(body_motions[index])
        train_body_motion_len.append(motion_len)

    test_body_motion_seqs = [] 
    test_body_motion_len = []
    for name in test_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames

        test_body_motion_seqs.append(body_motions[index])
        test_body_motion_len.append(motion_len)

    return train_body_motion_seqs, np.array(train_body_motion_len, dtype=int), test_body_motion_seqs, np.array(test_body_motion_len, dtype=int)


def load_train_motion_info_BEAT_large(data_dir, test_files, window_size, stride):
    # body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    # assert stride % ds_rate == 0

    body_motion_names = [] 
    body_motion_len = []
    
    train_files = list(set(config.file_info.keys()) - set(test_files))
    # train_files = list(set(config.file_info.keys()))
    for name in train_files:
        index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames

        body_motion_names.append(name)
        body_motion_len.append(motion_len)

    return body_motion_names, np.array(body_motion_len, dtype=int)


def load_test_motion_info_BEAT_large(data_dir, test_files, window_size, stride):
    # body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    assert stride == window_size

    body_motion_names = [] 
    body_motion_len = []
        
    # test_files = list(set(config.file_info.keys()))
    for name in test_files:
        motion_len = config.file_info[name].num_motion_frames

        body_motion_names.append(name)
        body_motion_len.append(motion_len)

    return body_motion_names, np.array(body_motion_len, dtype=int)


def load_train_info_BEAT_large(data_dir, test_files, window_size, stride):
    # body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    # assert stride % ds_rate == 0

    file_names = [] 
    body_motion_len = []
    audio_features_len = []
    
    train_files = list(set(config.file_info.keys()) - set(test_files))
    # train_files = list(set(config.file_info.keys()))
    for name in train_files:
        # index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames
        audio_len = config.file_info[name].num_audio_frames

        if motion_len > 0 and audio_len > 0:
            assert motion_len % audio_len == 0

            file_names.append(name)
            body_motion_len.append(motion_len)
            audio_features_len.append(audio_len)

    return file_names, np.array(body_motion_len, dtype=int), np.array(audio_features_len, dtype=int)


def load_test_info_BEAT_large(data_dir, test_files, window_size, stride):
    # body_motions = np.load(os.path.join(data_dir, "body_motions.npy"), allow_pickle=True)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = EasyDict(json.load(f))

    ds_rate = config.ds_rate
    assert window_size % ds_rate == 0
    assert stride == window_size

    file_names = [] 
    body_motion_len = []
    audio_features_len = []
        
    for name in test_files:
        # index = config.file_info[name].index
        motion_len = config.file_info[name].num_motion_frames
        audio_len = config.file_info[name].num_audio_frames
        
        if motion_len > 0 and audio_len > 0:
            file_names.append(name)
            body_motion_len.append(motion_len)
            audio_features_len.append(audio_len)

    return file_names, np.array(body_motion_len, dtype=int), np.array(audio_features_len, dtype=int)


def to_bvh(body_motion, config, save_path):
    import pickle
    from .BVH_loader import save as bvh_save
    from .BVH_loader import load as bvh_load

    with open(os.path.join(config.data.dir, 'motion_data_template.pkl'), 'rb') as f:
        data = pickle.load(f)

    data._num_frames = len(body_motion)

    if hasattr(config, "structure"):
        joint_channel = config.structure.joint_channel

    elif hasattr(config, "structure_vqvae"):
        joint_channel = config.structure_vqvae.joint_channel

    else:
        raise AttributeError
    
    body_motion = body_motion.reshape(data.num_frames, -1, joint_channel)
    t, j, c = body_motion.shape

    pre_root_pos_vel = body_motion[:, -1, 0:3]
    root_pos = np.cumsum(pre_root_pos_vel, axis = 0)

    rot = MathHelper.exp_to_quat(body_motion[:, :-1, :])
    rot = MathHelper.flip_quat_arr_by_dot(rot)
    data._joint_rotation = np.array(rot[0])

    data._joint_translation = np.zeros((data.num_frames, data.num_joints, 3))

    data._joint_translation[:, 0, :] = root_pos  # HARDCODE
    
    data._joint_position = None
    data._joint_orientation = None
    data.align_joint_rotation_representation()
    data.recompute_joint_global_info()

    bvh_save(data, save_path)


def visualize_and_write(results, gts, config, vis_dir, epoch, need_gt_pos=1, quants=None, files_name=None):
    if config.data.name == 'BEAT' or config.data.name == 'zeroeggs' or config.data.name == 'mocap':
        body_scaler = load_standard_scalers(os.path.join(config.data.dir, "body_scaler.sav"))
        for i in range(len(results)):
            body_motion_gt = gts[i][0].data.cpu().numpy()  # time, 6J
            body_motion_gt = body_scaler.inverse_transform(body_motion_gt)
            # body_motion_gt = add_root_offset_BEAT(body_motion_gt)

            body_motion = results[i][0].data.cpu().numpy()  # time, 6J
            print(body_motion.shape)
            body_motion = body_scaler.inverse_transform(body_motion)
            # body_motion = add_root_offset_BEAT(body_motion)

            if files_name is not None:
                bvh_name = files_name[i]
            elif quants is not None:
                assert len(results) == len(quants.keys())
                bvh_name = f'epoch_{epoch}_' + list(quants.keys())[i]
            else:
                bvh_name = f'epoch_{epoch}_{i}'
            print(f'writing {bvh_name} to bvh')

            to_bvh(body_motion, config, os.path.join(vis_dir, 'BVH', bvh_name+'.bvh'))
            to_bvh(body_motion_gt, config, os.path.join(vis_dir, 'BVH', 'gt_' + bvh_name + '.bvh'))
    else:
        raise ValueError
    

def visualize_and_write_no_gt(results, config, vis_dir, epoch, need_gt_pos=1, files_name=None):
    if config.data.name == 'BEAT' or config.data.name == 'zeroeggs' or config.data.name == 'mocap':
        body_scaler = load_standard_scalers(os.path.join(config.data.dir, "body_scaler.sav"))
        for i in range(len(results)):
            body_motion = results[i][0].data.cpu().numpy()  # time, 6J
            print(body_motion.shape)
            body_motion = body_scaler.inverse_transform(body_motion)
            # body_motion = add_root_offset_BEAT(body_motion)

            if files_name is not None:
                bvh_name = files_name[i]

            print(f'writing {bvh_name} to bvh')

            to_bvh(body_motion, config, os.path.join(vis_dir, 'BVH', bvh_name+'.bvh'))
    else:
        raise ValueError
    

def visualize_and_write_for_single(results, config, vis_dir, epoch, file_name=None):
    if config.data.name == 'BEAT' or config.data.name == 'zeroeggs' or config.data.name == 'mocap':
        body_scaler = load_standard_scalers(os.path.join(config.data.dir, "body_scaler.sav"))
        for i in range(len(results)):

            body_motion = results[i][0].data.cpu().numpy()  # time, 6J
            # print(body_motion.shape)
            # print(body_motion)
            if len(body_motion.shape) >= 3:
                body_motion = body_motion[0]
            body_motion = body_scaler.inverse_transform(body_motion)
            # body_motion = add_root_offset_BEAT(body_motion)

            if file_name:
                bvh_name = file_name
            else:
                bvh_name = f'epoch_{epoch}_{i}'
                
            print(f'writing {bvh_name} to bvh')

            to_bvh(body_motion, config, os.path.join(vis_dir, 'BVH', bvh_name+'.bvh'))
    else:
        raise ValueError