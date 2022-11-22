import torch
import time
import timeit
import zmq
import numpy as np
import os
import scipy.ndimage as filters
from trainer import Trainer
from motion.Quaternions import Quaternions
import motion.Animation as Animation
from etc.output2bvh import compute_posture
import motion.BVH as BVH
from motion.Pivots import Pivots
from preprocess.generate_dataset import process_data
from etc.utils import ensure_dirs, get_config


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def initialize_path(config):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    ensure_dirs([config['main_dir'], config['model_dir']])


def trajectory2string(tra):
    tra = np.array2string(tra, separator=',')
    return tra


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

frame = []
frame_cnt = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chosen_joints = np.array([
    0,
    2, 3, 4, 5,
    7, 8, 9, 10,
    12, 13, 15, 16,
    18, 19, 20, 22,
    25, 26, 27, 29])
parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19])

# initialize path
cfg = get_config('./model_ours/info/config.yaml')
initialize_path(cfg)

# Trainer
trainer = Trainer(cfg)
epochs = trainer.load_checkpoint()

# import norm
data_norm_dir = os.path.join(cfg['data_dir'], 'norm')
mean_path = os.path.join(data_norm_dir, "motion_mean.npy")
std_path = os.path.join(data_norm_dir, "motion_std.npy")
mean = np.load(mean_path, allow_pickle=True).astype(np.float32)
std = np.load(std_path, allow_pickle=True).astype(np.float32)
mean = mean[:, np.newaxis, np.newaxis]
std = std[:, np.newaxis, np.newaxis]

socket_output = "None"

current_sty_index = 0
sty_data_list = []
sty_data_file_list = ['./datasets/Xia/test_bvh/strutting_normal_walking_000.bvh',
                      './datasets/Xia/test_bvh/old_normal_walking_000.bvh',
                      './datasets/Xia/test_bvh/proud_normal_walking_000.bvh',
                      './datasets/Xia/test_bvh/childlike_normal_walking_000.bvh',
                      './datasets/Xia/test_bvh/depressed_normal_walking_000.bvh']
for i in range(len(sty_data_file_list)):
    sty_mot, sty_root, sty_fc = process_data(sty_data_file_list[i], divide=False)
    sty_mot, sty_root, sty_fc = sty_mot[0], sty_root[0], sty_fc[0]

    sty_motion_raw = np.transpose(sty_mot, (2, 1, 0))
    sty_motion = (sty_motion_raw - mean) / std
    sty_motion = torch.from_numpy(sty_motion[np.newaxis].astype('float32'))
    sty_data = sty_motion.to(device)
    sty_data_list.append(sty_data)

# ii = 0
while True:
    #  Wait for next request from client
    message = socket.recv()
    message = message.decode('utf-8')
    # print("Received request: %s" % message)

    if message != '':
        # add joint to frame
        joints = []
        data = message.split('|')
        current_sty_index = int(data[0])
        for value in data[1:]:
            joint = []
            data2 = value.split(',')
            for value2 in data2:
                if value2 != '':
                    joint.append(float(value2))
            if len(joint) > 0:
                joints.append(joint)
        frame.append(joints)

        # if collect enough frame
        if len(frame) > frame_cnt:
            del frame[0]
            motion = frame[:]
            motion = np.array(motion)

            # position = motion[:, :, :3]

            # """ Put on Floor """
            # fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
            # foot_heights = np.minimum(position[:, fid_l, 1], position[:, fid_r, 1]).min(axis=1)
            # floor_height = softmin(foot_heights, softness=0.5, axis=0)
            # position[:, :, 1] -= floor_height
            # new_frame[:, :, 1] -= floor_height

            # """ Extract Forward Direction and smooth """
            # sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
            # across = (
            #         (position[:, sdr_l] - position[:, sdr_r]) +
            #         (position[:, hip_l] - position[:, hip_r])
            # )
            # across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
            #
            # direction_filterwidth = 20
            # forward = filters.gaussian_filter1d(
            #     np.cross(across[:, :3], np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
            # forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
            #
            # target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
            # root_rotation = Quaternions.between(forward, target)[:, np.newaxis]
            #
            # root_velocity = root_rotation[:-1] * (position[1:, 0:1] - position[:-1, 0:1])
            # root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps  # to angle-axis
            # root_full = np.concatenate([root_velocity[:, :, 0:1], root_velocity[:, :, 2:3], np.expand_dims(root_rvelocity, axis=-1)], axis=-1)
            # root_full = np.append(root_full, [root_full[frame_cnt-2]], axis=0)

            # """ Foot Contacts """
            # fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
            # velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])
            # feet_l_x = (position[1:, fid_l, 0] - position[:-1, fid_l, 0]) ** 2
            # feet_l_y = (position[1:, fid_l, 1] - position[:-1, fid_l, 1]) ** 2
            # feet_l_z = (position[1:, fid_l, 2] - position[:-1, fid_l, 2]) ** 2
            # feet_l_h = position[:-1, fid_l, 1]
            # feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)
            #           & (feet_l_h < heightfactor)).astype(np.float)
            #
            # feet_r_x = (position[1:, fid_r, 0] - position[:-1, fid_r, 0]) ** 2
            # feet_r_y = (position[1:, fid_r, 1] - position[:-1, fid_r, 1]) ** 2
            # feet_r_z = (position[1:, fid_r, 2] - position[:-1, fid_r, 2]) ** 2
            # feet_r_h = position[:-1, fid_r, 1]
            # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)
            #           & (feet_r_h < heightfactor)).astype(np.float)
            #
            # foot_contacts = np.concatenate([feet_l, feet_r], axis=-1).astype(np.int32)

            # normalization
            cnt_motion_raw = np.transpose(motion, (2, 1, 0))
            cnt_motion = (cnt_motion_raw - mean) / std

            cnt_motion = torch.from_numpy(cnt_motion[np.newaxis].astype('float32'))  # (1, dim, joint, seq)

            loss_test = {}
            with torch.no_grad():
                start_t = timeit.default_timer()
                cnt_data = cnt_motion.to(device)

                outputs = trainer.test(cnt_data, sty_data_list[current_sty_index])
                output = outputs["stylized"].squeeze()
                output = output.cpu().numpy() * std + mean
                output = np.swapaxes(output, 0, 2)
                socket_output = "not none"
                # ii += 1

                terminate_t = timeit.default_timer()
                FPS = (terminate_t - start_t)
                print(FPS)

            # root = motion[:,0,0:3]
            # root = np.reshape(root, (frame_cnt, 1, 3))
            # cnt_mot, cnt_root, cnt_fc = \
            #     process_data("./datasets/cmu/test_bvh/127_21.bvh", divide=False)
            # root = cnt_root[0]
            #
            # # BUILD BVH FILE
            # if ii == 100:
            #     content_bvh_file = "./datasets/cmu/test_bvh/127_21.bvh"
            #     rest, names, _ = BVH.load(content_bvh_file)
            #     names = np.array(names)
            #     names = names[chosen_joints].tolist()
            #     offsets = rest.copy().offsets[chosen_joints]
            #     orients = Quaternions.id(len(parents))
            #
            #     new_frame = np.swapaxes(output, 0, 2)
            #     print(new_frame.shape)
            #     motion = compute_posture(new_frame, root)
            #
            #     local_joint_xforms = motion['local_joint_xforms']
            #
            #     s = local_joint_xforms.shape[:2]
            #     rotations = Quaternions.id(s)
            #     for f in range(s[0]):
            #         for j in range(s[1]):
            #             rotations[f, j] = Quaternions.from_transforms2(local_joint_xforms[f, j])
            #
            #     positions = offsets[np.newaxis].repeat(len(rotations), axis=0)
            #     positions[:, 0:1] = motion['positions'][:, 0:1]
            #
            #     anim = Animation.Animation(rotations, positions,
            #                                orients, offsets, parents)
            #
            #     file_path = os.path.join("./output/", "output.bvh")
            #     BVH.save(file_path, anim, names, frametime=1.0 / 30.0)
            #     print("finish")

    #  Do some 'work'.
    #  Try reducing sleep time to 0.01 to see how blazingly fast it communicates
    #  In the real world usage, you just need to replace time.sleep() with
    #  whatever work you want python to do, maybe a machine learning task?
    time.sleep(1 / frame_cnt)

    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    if not socket_output == "None":
        socket_output = trajectory2string(output[output.shape[0] - 1, :, :]).replace('[', '').replace(']', '').replace(' ', '')
        socket.send_string(socket_output)
        socket_output = "None"
    else:
        socket.send(b"")
