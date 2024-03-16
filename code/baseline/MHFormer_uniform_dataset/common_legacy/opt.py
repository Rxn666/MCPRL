import argparse
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=512, type=int)
        self.parser.add_argument('--d_hid', default=1024, type=int)
        self.parser.add_argument('--dataset', type=str, default='mpii')
        self.parser.add_argument('-k', '--keypoints', default='gt', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', default=1)
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=30)
        self.parser.add_argument('--batch_size', type=int, default=1024, help='can be changed depending on your machine')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=27)
        self.parser.add_argument('--pad', type=int, default=0)
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--local-rank', help='local rank for dist')

    def parse(self):
        self.init()
        
        self.opt = self.parser.parse_args()

        if self.opt.test:
            self.opt.train = 0
            
        self.opt.pad = (self.opt.frames-1) // 2

        # self.opt.subjects_train = 'S100,S101,S102,S103,S104,S105,S106,S107,S108,S109,S110,S111,S112,S113,S114,S115,S116,S117,S118,S119,S120,S121,S122,S123,S124,S125,S126,S127,S128,S129,S130,S131,S132,S133,S134,S135,S136,S137,S138,S139,S140,S141,S142,S143,S144,S145,S146,S147,S148,S149,S150,S151,S152,S153,S154,S155,S156,S157,S158,S159,S160'
        # self.opt.subjects_test = 'S9,S11'
        self.opt.subjects_train = 'S200,S201,S202,S203,S204,S205,S206,S207,S208,S209,S210,S211,S212,S213,S214,S215'
        self.opt.subjects_test = 'S216,S217,S218,S219,S220,S221'

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')
            self.opt.checkpoint = 'checkpoint/' + logtime + '%d'%(self.opt.frames)
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
       
        return self.opt





        
