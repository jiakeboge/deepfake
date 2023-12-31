#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from .data.config import cfg
from .pyramidbox import build_net
from torch.autograd import Variable
from .utils.augmentations import to_chw_bgr


class Config:
    def __init__(self, json):
        self.__dict__.update(json)


def detect_face(net, img, thresh, use_cuda):
    height, width, _ = img.shape
    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    bboxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            box = []
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            j += 1
            box += [pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1], score]
            bboxes += [box]

    return bboxes


def test(callback, args):
    args = Config(args)
    args.epoch = -1
    args.thresh = 0.01
    args.cuda = True
    args.model_path = './AiServer/Pyramidbox/weights'
    args.model = os.path.join(args.model_path, f'pyramidbox_{args.epoch}.pth')
    args.save_path = os.path.join(cfg.FACE.FDDB_DIR, f'pyramidbox_resnet152_{args.epoch}')

    if args.use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    FDDB_IMG_DIR = os.path.join(cfg.FACE.FDDB_DIR, 'fddb_images')
    FDDB_FOLD_DIR = os.path.join(cfg.FACE.FDDB_DIR, 'FDDB-folds')
    FDDB_RESULT_DIR = args.save_path
    FDDB_RESULT_IMG_DIR = os.path.join(FDDB_RESULT_DIR, 'images')

    if not os.path.exists(FDDB_RESULT_IMG_DIR):
        os.makedirs(FDDB_RESULT_IMG_DIR)

    net = build_net('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if args.use_cuda:
        net.cuda()
        cudnn.benckmark = True

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    counter = 0

    for i in range(10):
        txt_in = os.path.join(FDDB_FOLD_DIR, 'FDDB-fold-%02d.txt' % (i + 1))
        txt_out = os.path.join(FDDB_RESULT_DIR, 'fold-%02d-out.txt' % (i + 1))
        answer_in = os.path.join(
            FDDB_FOLD_DIR, 'FDDB-fold-%02d-ellipseList.txt' % (i + 1))
        with open(txt_in, 'r') as fr:
            lines = fr.readlines()
        fout = open(txt_out, 'w')
        ain = open(answer_in, 'r')
        for line in lines:
            line = line.strip()
            img_file = os.path.join(FDDB_IMG_DIR, line + '.jpg')
            out_file = os.path.join(
                FDDB_RESULT_IMG_DIR, line.replace('/', '_') + '.jpg')
            counter += 1
            t1 = time.time()
            # img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = Image.open(img_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)
            bboxes = detect_face(net, img, args.thresh, args.use_cuda)
            t2 = time.time()
            print('Detect %04d th image costs %.4f' % (counter, t2 - t1))
            fout.write('%s\n' % line)
            fout.write('%d\n' % len(bboxes))
            for bbox in bboxes:
                x1, y1, w, h, score = bbox
                fout.write('%d %d %d %d %lf\n' % (x1, y1, w, h, score))
            ain.readline()
            n = int(ain.readline().strip())
            for i in range(n):
                line = ain.readline().strip()
                line_data = [float(_) for _ in line.split(' ')[:5]]
                major_axis_radius, minor_axis_radius, angle, center_x, center_y = line_data
                angle = angle / 3.1415926 * 180.
                center_x, center_y = int(center_x), int(center_y)
                major_axis_radius, minor_axis_radius = int(
                    major_axis_radius), int(minor_axis_radius)
                cv2.ellipse(img, (center_x, center_y), (major_axis_radius,
                                                        minor_axis_radius), angle, 0, 360, (255, 0, 0), 2)

            for bbox in bboxes:
                x1, y1, w, h, score = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(out_file, img)
        fout.close()
        ain.close()


def test_customize(callback, args):
    args = Config(args)
    args.epoch = -1
    args.thresh = 0.01
    # args.model_path = 'checkpoint/pyramidbox_-1.pth'
    args.model = args.model_path #os.path.join(args.model_path, f'pyramidbox_{args.epoch}.pth')
    # args.save_path = os.path.join(args.output_path, f'pyramidbox_resnet152_{args.epoch}')

    args.save_path = args.input_path.replace("media", "export")
    if args.use_cuda:
        device = "cuda:0"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
        torch.set_default_tensor_type('torch.FloatTensor')

    try:
        net = build_net('test', cfg.NUM_CLASSES, res=50)
        net.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
    except RuntimeError:
        net = build_net('test', cfg.NUM_CLASSES, res=152)
        net.load_state_dict(torch.load(args.model, map_location=torch.device(device)))

    net.eval()

    if args.use_cuda:
        net.cuda()
        cudnn.benckmark = True

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    # for root, _, videos in os.walk(args.input_path):
    # for video in videos:
    read_frame_counter = -1
    save_frame_counter = 0
    # video_path = os.path.join(root, video)
    video_path = args.input_path
    video_name = video_path.split("/")[-1]
    frame_file_save_path = os.path.join(args.save_path)
    if not os.path.exists(frame_file_save_path):
        os.makedirs(frame_file_save_path)

    print('Detect %s' % (video_path))
    cap = cv2.VideoCapture(video_path)
    txt_out = os.path.join(frame_file_save_path, 'fold-out.txt')
    fout = open(txt_out, 'w')
    result_dict = {}
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        read_frame_counter += 1
        if read_frame_counter % args.interval == 0:

            save_frame_counter += 1
            cv2.imwrite(os.path.join(frame_file_save_path, str(save_frame_counter)+".jpg"), frame)
            img = np.array(frame)
            bboxes = detect_face(net, img, args.thresh, args.use_cuda)
            print('Detect %04d th image' % (save_frame_counter))
            fout.write('%s\n' % (str(save_frame_counter) + ".jpg"))
            fout.write('%d\n' % len(bboxes))
            f_boxes = []
            for bbox in bboxes:
                x1, y1, w, h, score = bbox
                f_boxes.append([x1/frame.shape[1], y1/frame.shape[0], w/frame.shape[1], h/frame.shape[0]])
                fout.write('%d %d %d %d %lf\n' % (x1, y1, w, h, score))
            #for bbox in bboxes:
            #    x1, y1, w, h, score = bbox
            #    x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
            #    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #    cv2.imwrite(os.path.join(frame_file_save_path, str(save_frame_counter) + "_detected.jpg"), img)
            result_dict["frame_{}".format(read_frame_counter)] = f_boxes
    print(result_dict)
    fout.close()
    return result_dict


if __name__ == '__main__':
    test()
