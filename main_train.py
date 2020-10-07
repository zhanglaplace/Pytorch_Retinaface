#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangfeng05
@license: (C) Copyright 2020-2032 .
@contact: zhangfeng05@kuaishou.com
@file: main_train.py
@time: 2020-10-07 13:18
@desc: 修改为基于amp的distribute的训练
"""
import yaml
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.retinaface import RetinaFace
from utils.datasets import *
from utils.utils import *
from torch.cuda import amp

wdir = 'weights' + os.sep  # weights dir
os.makedirs(wdir, exist_ok=True)
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

logger = logging.getLogger(__name__)


def train(opt, train_dict, device, tb_writer=None):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser("--retinaface for face detect")
    parser.add_argument('--cfg', type=str, default='exp/retinaface_resnet50.yaml', help='*.cfg path')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    with open(opt.cfg) as f:
        train_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_dict['weight'] = last if train_dict['pretrain'] and not os.path.exists(train_dict['weights']) else train_dict[
        'weight']
    device = select_device(train_dict['device'], batch_size=train_dict['batch_size'])

    # DDP mode
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert train_dict['batch_size'] % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        train_dict['batch_size'] = train_dict['batch_size'] // opt.world_size

    logger.info(opt)
    logger.info(data_dict)

    tb_writer = None
    if opt.global_rank in [-1, 0]:
        logger.info(
            'Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % train_dict['logdir'])
        tb_writer = SummaryWriter(
            log_dir=increment_dir(Path(train_dict['logdir']) / 'exp', train_dict['name']))  # runs/exp

    train(opt, train_dict, device, tb_writer)
