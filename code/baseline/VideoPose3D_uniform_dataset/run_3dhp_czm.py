import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from common.opt import opts

from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_3dhp_mae import Fusion
import scipy.io as scio
# from common.arguments import parse_args
from common.model import *
from common.loss import *


# args = parse_args()
# print(args)

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu




def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model = model

    if split == 'train':
        model.train()
    else:
        model.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()

    action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
    action_error_sum_MAE = define_error_list(actions)

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, 0)):

        if split == "train":
            batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
        else:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        gt_3D = gt_3D[:, 13]
        print(gt_3D.shape)

        # print(input_2D.shape)

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        # print(input_2D.shape)
        # torch.Size([1024, 27, 17, 2])

        # input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)

        # print(input_2D.shape)
        # output_3D, output_3D_VTE = model(input_2D)

        # output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)
        # output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)

        # output_3D_VTE = output_3D_VTE * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D_VTE.size(1),opt.out_joints, opt.out_channels)
        # output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.out_joints, opt.out_channels)
        # output_3D_single = output_3D

        print(input_2D.shape)
        output_3D = output_3D_single = output_3D_VTE = model(input_2D)
        print(output_3D.shape)

        if split == 'train':
            pred_out = output_3D_VTE

        elif split == 'test':
            pred_out = output_3D_single

        # input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints,2)


        loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(output_3D_single, out_target_single)


        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_out[:,:,14,:] = 0
            joint_error = mpjpe_cal(pred_out, out_target).item()

            error_sum.update(joint_error*N, N)

        elif split == 'test':

            pred_out[:, :, 14, :] = 0
                #action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, subject)
            joint_error_test = mpjpe_cal(pred_out, out_target).item()
            out = pred_out
                # if opt.refine:
                #     post_out[:, :, 14, :] = 0
                #     action_error_sum_post_out = test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset, subject)

            if opt.train == 0:
                for seq_cnt in range(len(seq)):
                    seq_name = seq[seq_cnt]
                    if seq_name in data_inference:
                        data_inference[seq_name] = np.concatenate(
                            (data_inference[seq_name], out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                    else:
                        data_inference[seq_name] = out[seq_cnt].permute(2, 1, 0).cpu().numpy()

            error_sum_test.update(joint_error_test * N, N)

    if split == 'train':
        return loss_all['loss'].avg, error_sum.avg

    elif split == 'test':
        #p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        if opt.train == 0:
            for seq_name in data_inference.keys():
                data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
            mat_path = os.path.join(opt.checkpoint, 'inference_data.mat')
            scio.savemat(mat_path, data_inference)

        return error_sum_test.avg



def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)   
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) 

    print(input_2D_flip.shape)
    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip)

    output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1

    output_3D_flip_VTE[:, :, :, joints_left + joints_right] = output_3D_flip_VTE[:, :, :, joints_right + joints_left]
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip)

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, output_3D_VTE

if __name__ == '__main__':
    opt.manualSeed = 1

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    actions = define_actions(opt.actions)

    if opt.train:
        #train_data = Fusion(opt=opt, train=True, root_path=root_path)
        train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        #test_data = Fusion(opt=opt, train=False,root_path =root_path)
        test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = 17

    model = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3]).cuda()

    # model = {}
    # model['trans'] = nn.DataParallel(Model(opt)).cuda()
    # model['refine'] = nn.DataParallel(refine(opt)).cuda()
    # model['MAE'] = nn.DataParallel(Model_MAE(opt)).cuda()


    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)


    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
        if opt.test == 1:
            p1 = val(opt, actions, test_dataloader, model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], 'no_refine')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('p1: %.2f' % (p1))
                break
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))
                print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay


