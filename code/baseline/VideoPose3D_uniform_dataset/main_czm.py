from common.arguments import parse_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import random
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

args = parse_args()
print(args)

def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip     = model(input_2D_flip)

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D


class KeyPointsDataset(Dataset):
    def __init__(self, npz_path, sequence_length, subjects, stride=1):
        # Load the dataset from the given npz file
        self.dataset = np.load(npz_path, allow_pickle=True)['dataset'].item()

        # Filter the dataset based on the provided subjects
        self.dataset = {subject: self.dataset[subject] for subject in subjects if subject in self.dataset}

        # Flatten the dataset for easy indexing
        self.flat_data = []
        for subject in self.dataset:
            for action in self.dataset[subject]:
                self.flat_data.append((subject, action, self.dataset[subject][action]['data_2d']))

        self.sequence_length = sequence_length
        self.stride = stride

        # Compute all possible subsequences using the sliding window approach
        self.all_sequences = []
        for idx, (subject, action, data_2d) in enumerate(self.flat_data):
            for start_idx in range(0, len(data_2d) - self.sequence_length + 1, self.stride):
                self.all_sequences.append((idx, start_idx))


    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        data_idx, start_idx = self.all_sequences[idx]
        subject, action, data_2d = self.flat_data[data_idx]

        # Extract the 2D sequence and the middle 3D frame
        data_2d_sequence = data_2d[start_idx:start_idx + self.sequence_length]
        middle_frame_idx = start_idx + self.sequence_length // 2
        data_3d_middle = self.dataset[subject][action]['data_3d'][middle_frame_idx]

        return {
            'data_2d': torch.tensor(data_2d_sequence, dtype=torch.float32),
            'data_3d': torch.tensor(data_3d_middle, dtype=torch.float32).unsqueeze(0)
            # Add an extra dimension to match the expected shape
        }


if __name__ == '__main__':
    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

    train_dataset = KeyPointsDataset(npz_path="./data/uniform_dataset.npz", sequence_length=27, subjects=subjects_train)
    test_dataset = KeyPointsDataset(npz_path="./data/uniform_dataset.npz", sequence_length=27, subjects=subjects_test)

    print(len(train_dataset))
    print(len(test_dataset))
    # print(train_dataset[100]['data_2d'].shape)
    # print(train_dataset[100]['data_3d'].shape)
    # print(test_dataset[100]['data_2d'].shape)
    # print(test_dataset[100]['data_3d'].shape)



    # Create data loaders
    batch_size = args.batch_size  # You can adjust this value based on your memory capacity
    num_workers = 8  # Number of subprocesses to use for data loading. Adjust based on your system's capabilities.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    model = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3], causal=args.causal,
                              dropout=args.dropout, channels=args.channels, dense=args.dense)
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = model.cuda()

    # Hyperparameters
    lr = args.learning_rate
    lr_decay = args.lr_decay
    initial_momentum = 0.1
    final_momentum = 0.001
    epochs = args.epochs

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    # Lists to store losses for visualization later
    losses_3d_train = []
    losses_3d_valid = []

    # Training loop
    for epoch in range(epochs):
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0

        model.train()
        for batch in tqdm(train_loader):
            inputs_2d = batch['data_2d']
            inputs_3d = batch['data_3d']

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)  # Assuming you have a function mpjpe to compute the loss
            epoch_loss_3d_train += loss_3d_pos.item() * inputs_3d.shape[0]
            N += inputs_3d.shape[0]

            loss_3d_pos.backward()
            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation on the validation set
        with torch.no_grad():
            model.eval()
            epoch_loss_3d_valid = 0
            N = 0
            for batch in test_loader:
                inputs_2d = batch['data_2d']
                inputs_3d = batch['data_3d']

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()

                # Predict 3D poses
                predicted_3d_pos = model(inputs_2d)
                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_valid += loss_3d_pos.item() * inputs_3d.shape[0]
                N += inputs_3d.shape[0]

            losses_3d_valid.append(epoch_loss_3d_valid / N)

        elapsed = (time() - start_time) / 60
        print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1] * 1000,
            0,  # Placeholder for losses_3d_train_eval, since it's not defined in your code
            losses_3d_valid[-1] * 1000
        ))

        # Decay learning rate
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Save checkpoints
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, f'epoch_{epoch}.bin')
            print(f'Saving checkpoint to {chk_path}')
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, chk_path)




