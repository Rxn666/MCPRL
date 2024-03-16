
import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_3dhp import ChunkedGenerator

class Fusion(data.Dataset):
    def __init__(self, opt, root_path, train=True, MAE=False):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.MAE = MAE
        if self.train:
            self.poses_train, self.poses_train_2d = self.prepare_data(opt.root_path, train=True)
            # self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
            #                                                                        subset=self.subset)
            # print(self.poses_train.keys())
            # print(self.poses_train_2d.keys())
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, None, self.poses_train,
                                              self.poses_train_2d, None, chunk_length=self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all, MAE=MAE, train=True)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.poses_test, self.poses_test_2d, self.valid_frame = self.prepare_data(opt.root_path, train=False)
            # self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
            #                                                                     subset=self.subset)
            # for ts in self.valid_frame.keys():
            #     self.valid_frame[ts] = np.ones((self.valid_frame[ts].shape))
            # print(self.poses_test.keys())
            # print(self.poses_test_2d.keys())
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, None, self.poses_test,
                                              self.poses_test_2d, self.valid_frame,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, MAE=MAE, train=False)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, path, train=True):
        out_poses_3d = {}
        out_poses_2d = {}
        valid_frame = {}

        self.kps_left, self.kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        self.joints_left, self.joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]

        if train == True:
            data_3d = np.load(path+"data_train_uniform_65_80.npz", allow_pickle=True)['train_3d'].item()
            data_2d = np.load(path + "data_train_uniform_65_80.npz", allow_pickle=True)['train_2d'].item()

            for subject_name in data_3d.keys():
                out_poses_3d[(subject_name[0], subject_name[1], str(subject_name[2]))] = data_3d[subject_name]
                out_poses_2d[(subject_name[0], subject_name[1], str(subject_name[2]))] = data_2d[subject_name]

            # out_poses_3d = data_3d
            # out_poses_2d = data_2d

            return out_poses_3d, out_poses_2d
        else:
            data_3d = np.load(path + "data_test_uniform_65_80.npz", allow_pickle=True)['test_3d'].item()
            data_2d = np.load(path + "data_test_uniform_65_80.npz", allow_pickle=True)['test_2d'].item()

            # print()
            for subject_name in data_3d.keys():
                # if len(subject_name) > 1:
                #     print("3333333")
                if subject_name[0] == "T":
                    out_poses_3d[(subject_name, "TestAction", "0")] = data_3d[subject_name]
                    out_poses_2d[(subject_name, "TestAction", "0")] = data_2d[subject_name]
                else:
                    out_poses_3d[(subject_name[0], subject_name[1], str(subject_name[2]))] = data_3d[subject_name]
                    out_poses_2d[(subject_name[0], subject_name[1], str(subject_name[2]))] = data_2d[subject_name]
                    # out_poses_2d[(subject_name, "TestAction", 0)] = data_2d[subject_name]
                # print(subject_name)



            for ts in out_poses_3d.keys():
                valid_frame[ts] = np.ones((out_poses_3d[ts].shape[0]))
                # print(valid_frame[ts].shape)

            return out_poses_3d, out_poses_2d, valid_frame


    def __len__(self):
        return len(self.generator.pairs)
        #return 200

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        if self.MAE:
            cam, input_2D, seq, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip,
                                                                                      reverse)
            if self.train == False and self.test_aug:
                _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        else:
            cam, gt_3D, input_2D, seq, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

            if self.train == False and self.test_aug:
                _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float64(1.0)

        if self.MAE:
            if self.train == True:
                return cam, input_2D_update, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, input_2D_update, seq, scale, bb_box
        else:
            if self.train == True:
                return cam, gt_3D, input_2D_update, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, gt_3D, input_2D_update, seq, scale, bb_box



