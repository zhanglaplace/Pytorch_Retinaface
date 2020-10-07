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
import logging
import argparse
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from models.retinaface import RetinaFace
from utils.datasets import *
from utils.utils import *
from torch.cuda import amp

logger = logging.getLogger(__name__)


def train(opt, train_dict, device, tb_writer=None):
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(train_dict['logdir']) / 'logs'
    wdir = str(log_dir / 'weights') + os.sep
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = 'results.txt'
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(train_dict, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    rank = opt.global_rank
    init_seeds(2 + rank)
    train_path = train_dict['train']
    test_path = train_dict['val']
    train_dict['weights'] = last if not train_dict['pretrain'] or (
            train_dict['pretrain'] and not os.path.exists(train_dict['weight'])) else train_dict['weight']
    model = RetinaFace(train_dict, phase='Train')
    pretrained = False
    if os.path.exists(train_dict['weights']):
        pretrained = True
        logger('Loading resume network from ====>{}'.format(train_dict['weights']))
        state_dict = torch.load(train_dict['weight'], map_location=device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if train_dict['adam']:
        optimizer = optim.Adam(pg0, lr=train_dict['lr0'],
                               betas=(train_dict['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=train_dict['lr0'], momentum=train_dict['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': train_dict['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    epoch = train_dict['epoch']
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if state_dict['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = state_dict['best_fitness']

        # Results
        if state_dict.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(state_dict['training_results'])  # write results.txt

        # Epochs
        start_epoch = state_dict['epoch'] + 1
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, state_dict['epoch'], epochs))
            epochs += state_dict['epoch']  # finetune additional epochs

        del ckpt, state_dict


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
    logger.info(train_dict)

    tb_writer = None
    if opt.global_rank in [-1, 0]:
        logger.info(
            'Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % train_dict['logdir'])
        tb_writer = SummaryWriter(
            log_dir=Path(train_dict['logdir']) / 'exp')  # runs/exp

    train(opt, train_dict, device, tb_writer)
