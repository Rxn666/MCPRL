# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib
from torch.autograd import Variable


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject, MAE=False):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * batch_num * frame_num,
                                                   batch_num * frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item() * frame_num, frame_num)

    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = torch.mean(target, dim=1, keepdim=True)
    muY = torch.mean(predicted, dim=1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(1, 2), keepdim=True))

    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(1, 2), Y0)
    U, s, Vt = torch.svd(H)
    V = Vt.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))

    sign_detR = torch.sign(torch.det(R).unsqueeze(1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = torch.matmul(V, U.transpose(1, 2))

    tr = s.sum(dim=1, keepdim=True).unsqueeze(2)

    a = tr * normX / normY
    t = muX - a * torch.matmul(muY, R)

    predicted_aligned = a * torch.matmul(predicted, R) + t

    return torch.mean(torch.norm(predicted_aligned - target, dim=predicted.ndimension() - 1), dim=predicted.ndimension() - 2)



def define_actions(action):
    # actions = ["Directions","Discussion","Eating","Greeting", "Phoning","Photo","Posing","Purchases", "Sitting","SittingDown","Smoking","Waiting", "WalkDog","Walking","WalkTogether"]
    actions = ['TestAction']
    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1': AccumLoss(), 'p2': AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                                                    mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, model, model_name):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))

    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)
    return previous_name


def save_model_new(save_dir, epoch, data_threshold, lr, optimizer, model, model_name):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    # torch.save(model.state_dict(),
    #            '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    torch.save({
        'epoch': epoch,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model.state_dict(),
    },
        '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
