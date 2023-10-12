#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data

import os
import time
import torch
import argparse

import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from .data.config import cfg
from .pyramidbox import build_net
from .layers.modules import MultiBoxLoss
from .data.widerface import WIDERDetection, detection_collate

min_loss = np.inf


class Config:
    def __init__(self, json):
        self.__dict__.update(json)


def train(callback, args):
    args = Config(args)
    # args.epoch

    if not cfg.MULTIGPU:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if torch.cuda.is_available():
        if cfg.CUDA:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not cfg.CUDA:
            callback("WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(cfg.SAVE_FOLDER):
        os.makedirs(cfg.SAVE_FOLDER)

    train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

    train_loader = data.DataLoader(train_dataset, cfg.BATCH_SIZE,
                                   num_workers=0,#cfg.NUM_WORKERS,
                                   shuffle=True,
                                   collate_fn=detection_collate,
                                   pin_memory=True)

    iteration = 0
    start_epoch = 0
    step_index = 0
    per_epoch_size = len(train_dataset) // cfg.BATCH_SIZE

    pyramidbox = build_net('train', cfg.NUM_CLASSES)
    net = pyramidbox
    if cfg.RESUME:
        callback('Resuming training, loading {}...'.format(cfg.RESUME))
        start_epoch = net.load_weights(cfg.RESUME)
        iteration = start_epoch * per_epoch_size
    else:
        '''
        vgg_weights = torch.load(cfg.SAVE_FOLDER + args.basenet)
        callback('Load base network....')
        net.vgg.load_state_dict(vgg_weights)
        '''
        callback('Just load with imagenet pretrained...')

    if cfg.CUDA:
        net = net.cuda()
        if cfg.MULTIGPU:
            net = torch.nn.DataParallel(pyramidbox)
        cudnn.benckmark = True

    if not cfg.RESUME:
        callback('Initializing weights...')
        pyramidbox.extras.apply(pyramidbox.weights_init)
        pyramidbox.lfpn_topdown.apply(pyramidbox.weights_init)
        pyramidbox.lfpn_later.apply(pyramidbox.weights_init)
        pyramidbox.cpm.apply(pyramidbox.weights_init)
        pyramidbox.loc_layers.apply(pyramidbox.weights_init)
        pyramidbox.conf_layers.apply(pyramidbox.weights_init)

    if args.optim == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=cfg.WEIGHT_DECAY)
    elif args.optim == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    elif args.optim == "rmsprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError

    criterion1 = MultiBoxLoss(cfg, cfg.CUDA)
    criterion2 = MultiBoxLoss(cfg, cfg.CUDA, use_head_loss=True)
    callback('Loading wider dataset...')
    callback('Using the specified args:')
    # callback(args)
    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, cfg.GAMMA, step_index, args)

    net.train()

    for epoch in range(start_epoch, args.epochs):
        losses = 0
        for batch_idx, (images, face_targets, head_targets) in enumerate(train_loader):
            if cfg.CUDA:
                images = Variable(images.cuda())
                images = images.cuda()
                face_targets = [Variable(ann.cuda(), volatile=True)
                                for ann in face_targets]
                head_targets = [Variable(ann.cuda(), volatile=True)
                                for ann in head_targets]
            else:
                images = Variable(images)
                face_targets = [Variable(ann, volatile=True)
                                for ann in face_targets]
                head_targets = [Variable(ann, volatile=True)
                                for ann in head_targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, cfg.GAMMA, step_index, args)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            face_loss_l, face_loss_c = criterion1(out, face_targets)
            head_loss_l, head_loss_c = criterion2(out, head_targets)
            loss = face_loss_l + face_loss_c + head_loss_l + head_loss_c
            losses += loss.data.item()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            face_loss = (face_loss_l + face_loss_c).data.item()
            head_loss = (head_loss_l + head_loss_c).data.item()

            if iteration % 10 == 0:
                loss_ = losses / (batch_idx + 1)
                callback('Timer: {:.4f} sec.'.format(t1 - t0))
                callback('epoch ' + repr(epoch) + ' iter ' + repr(iteration) + ' || Loss:%.4f' % (loss_))
                callback('->> face Loss: {:.4f} || head loss : {:.4f}'.format(
                    face_loss, head_loss))
                callback('->> lr: {}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                callback('Saving state, iter:', iteration)
                file = 'pyramidbox_' + repr(iteration) + '.pth'
                torch.save(pyramidbox.state_dict(),
                           os.path.join(cfg.SAVE_FOLDER, file))
            iteration += 1

        val(epoch, net, pyramidbox, criterion1, criterion2, callback, args)


def val(epoch,
        net,
        pyramidbox,
        criterion1,
        criterion2,
        callback,
        args):
    val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
    val_batchsize = args.batch_size // 2
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                 num_workers=cfg.NUM_WORKERS,
                                 shuffle=False,
                                 collate_fn=detection_collate,
                                 pin_memory=True)
    net.eval()
    face_losses = 0
    head_losses = 0
    step = 0
    t1 = time.time()
    for batch_idx, (images, face_targets, head_targets) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            face_targets = [Variable(ann.cuda(), volatile=True)
                            for ann in face_targets]
            head_targets = [Variable(ann.cuda(), volatile=True)
                            for ann in head_targets]
        else:
            images = Variable(images)
            face_targets = [Variable(ann, volatile=True)
                            for ann in face_targets]
            head_targets = [Variable(ann, volatile=True)
                            for ann in head_targets]

        out = net(images)
        face_loss_l, face_loss_c = criterion1(out, face_targets)
        head_loss_l, head_loss_c = criterion2(out, head_targets)

        face_losses += (face_loss_l + face_loss_c).data.item()
        head_losses += (head_loss_l + head_loss_c).data.item()
        step += 1

    tloss = face_losses / step

    t2 = time.time()
    callback('test Timer:{:.4f} .sec'.format(t2 - t1))
    callback('epoch ' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        callback('Saving best state,epoch', epoch)
        torch.save(pyramidbox.state_dict(), os.path.join(
            cfg.SAVE_FOLDER, 'pyramidbox.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': pyramidbox.state_dict(),
    }
    torch.save(states, os.path.join(
        cfg.SAVE_FOLDER, 'pyramidbox_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step, args):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
