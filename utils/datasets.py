#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhangfeng05
@license: (C) Copyright 2020-2032 .
@contact: zhangfeng05@kuaishou.com
@file: datasets.py
@time: 2020-10-07 13:22
@desc:
"""

import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

#
# class LoadImagesAndLabels(Dataset):  # for training/testing
#     def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
#                  cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
#         try:
#             self.f = []  # image files
#             print(path)
#             p = str(Path(path))  # os-agnostic
#             if os.path.isfile(p):  # file
#                 with open(p, 'r') as t:
#                     for line in t:
#                         datas = line.strip().split('\t')
#                         img_path = datas[0]
#                         lab = json.loads(datas[1]) # lab 是[[x y w h],[x,y,w,h]]
#                         self.f.append([img_path,lab])
#             else:
#                 raise Exception('%s does not exist' % p)
#         except Exception as e:
#             raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))
#
#         self.image_list = [ele[0] for ele in self.f]
#         n = len(self.img_files)
#         assert n > 0, 'No images found in %s. See %s' % (path, help_url)
#         bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
#         nb = bi[-1] + 1  # number of batches
#
#         self.n = n  # number of images
#         self.batch = bi  # batch index of image
#         self.img_size = img_size
#         self.augment = augment
#         self.hyp = hyp
#         self.image_weights = image_weights
#         self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
#         self.mosaic_border = [-img_size // 2, -img_size // 2]
#         self.stride = stride
#
#         cache_path = path[:-4] + ".cache"
#         print(cache_path)
#         if os.path.isfile(cache_path):
#             cache = torch.load(cache_path)  # load
#             if cache['hash'] != get_hash(self.f):  # dataset changed
#                 cache = self.cache_labels(cache_path)  # re-cache
#         else:
#             cache = self.cache_labels(cache_path)  # cache
#
#         # Get labels
#         labels, shapes = zip(*[cache[x] for x in self.img_files])
#         self.shapes = np.array(shapes, dtype=np.float64)
#         self.labels = list(labels)
#
#         # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
#         if self.rect:
#             # Sort by aspect ratio
#             s = self.shapes  # wh
#             ar = s[:, 1] / s[:, 0]  # aspect ratio
#             irect = ar.argsort()
#             self.img_files = [self.img_files[i] for i in irect]
#             self.label_files = [self.label_files[i] for i in irect]
#             self.labels = [self.labels[i] for i in irect]
#             self.shapes = s[irect]  # wh
#             ar = ar[irect]
#
#             # Set training image shapes
#             shapes = [[1, 1]] * nb
#             for i in range(nb):
#                 ari = ar[bi == i]
#                 mini, maxi = ari.min(), ari.max()
#                 if maxi < 1:
#                     shapes[i] = [maxi, 1]
#                 elif mini > 1:
#                     shapes[i] = [1, 1 / mini]
#
#             self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
#
#         # Cache labels
#         create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
#         nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
#         pbar = enumerate(self.label_files)
#         if rank in [-1, 0]:
#             pbar = tqdm(pbar)
#         for i, file in pbar:
#             l = self.labels[i]  # label
#             for t in range(l.shape[0]):
#                 l[t][3] = min(1.0,l[t][3])
#                 l[t][4] = min(1.0,l[t][4])
#             if l is not None and l.shape[0]:
#                 assert l.shape[1] == 5, '> 5 label columns: %s' % file
#                 assert (l >= 0).all(), 'negative labels: %s' % file
#                 assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
#                 if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
#                     nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
#                 if single_cls:
#                     l[:, 0] = 0  # force dataset into single-class mode
#                 self.labels[i] = l
#                 nf += 1  # file found
#
#                 # Create subdataset (a smaller dataset)
#                 if create_datasubset and ns < 1E4:
#                     if ns == 0:
#                         create_folder(path='./datasubset')
#                         os.makedirs('./datasubset/images')
#                     exclude_classes = 43
#                     if exclude_classes not in l[:, 0]:
#                         ns += 1
#                         # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
#                         with open('./datasubset/images.txt', 'a') as f:
#                             f.write(self.img_files[i] + '\n')
#
#                 # Extract object detection boxes for a second stage classifier
#                 if extract_bounding_boxes:
#                     p = Path(self.img_files[i])
#                     img = cv2.imread(str(p))
#                     h, w = img.shape[:2]
#                     for j, x in enumerate(l):
#                         f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
#                         if not os.path.exists(Path(f).parent):
#                             os.makedirs(Path(f).parent)  # make new output folder
#
#                         b = x[1:] * [w, h, w, h]  # box
#                         b[2:] = b[2:].max()  # rectangle to square
#                         b[2:] = b[2:] * 1.3 + 30  # pad
#                         b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
#
#                         b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
#                         b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
#                         assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
#             else:
#                 ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
#                 # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove
#
#             if rank in [-1,0]:
#                 pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
#                     cache_path, nf, nm, ne, nd, n)
#         if nf == 0:
#             s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
#             print(s)
#             assert not augment, '%s. Can not train without labels.' % s
#
#         # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
#         self.imgs = [None] * n
#         if cache_images:
#             gb = 0  # Gigabytes of cached images
#             pbar = tqdm(range(len(self.img_files)), desc='Caching images')
#             self.img_hw0, self.img_hw = [None] * n, [None] * n
#             for i in pbar:  # max 10k images
#                 self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
#                 gb += self.imgs[i].nbytes
#                 pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)
#
#     def get_image_list(self,dir: str, test_image_list):
#         r"""get image list recursive"""
#         if os.path.isfile(dir):
#             test_image_list.append(dir)
#         elif os.path.isdir(dir):
#             for ele in os.listdir(dir):
#                 cur_path = os.path.join(dir, ele)
#                 self.get_image_list(cur_path, test_image_list)
#         return test_image_list
#
#     def cache_labels(self, path='labels.cache'):
#         # Cache dataset labels, check images and read shapes
#         x = {}  # dict
#         pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
#         for (img, label) in pbar:
#             try:
#                 l = []
#                 image = Image.open(img)
#                 image.verify()  # PIL verify
#                 # _ = io.imread(img)  # skimage verify (from skimage import io)
#                 shape = exif_size(image)  # image size
#                 assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
#                 if os.path.isfile(label):
#                     with open(label, 'r') as f:
#                         l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
#                 if len(l) == 0:
#                     l = np.zeros((0, 5), dtype=np.float32)
#                 x[img] = [l, shape]
#             except Exception as e:
#                 x[img] = [None, None]
#                 print('WARNING: %s: %s' % (img, e))
#
#         x['hash'] = get_hash(self.label_files + self.img_files)
#         torch.save(x, path)  # save for next time
#         return x
#
#     def __len__(self):
#         return len(self.img_files)
#
#     # def __iter__(self):
#     #     self.count = -1
#     #     print('ran dataset iter')
#     #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
#     #     return self
#
#     def __getitem__(self, index):
#         if self.image_weights:
#             index = self.indices[index]
#
#         hyp = self.hyp
#         if self.mosaic:
#             # Load mosaic
#             img, labels = load_mosaic(self, index)
#             shapes = None
#
#             # MixUp https://arxiv.org/pdf/1710.09412.pdf
#             if random.random() < hyp['mixup']:
#                 img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
#                 r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
#                 img = (img * r + img2 * (1 - r)).astype(np.uint8)
#                 labels = np.concatenate((labels, labels2), 0)
#
#         else:
#             # Load image
#             img, (h0, w0), (h, w) = load_image(self, index)
#
#             # Letterbox
#             shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
#             img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
#             shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
#
#             # Load labels
#             labels = []
#             x = self.labels[index]
#             if x.size > 0:
#                 # Normalized xywh to pixel xyxy format
#                 labels = x.copy()
#                 labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
#                 labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
#                 labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
#                 labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
#
#         if self.augment:
#             # Augment imagespace
#             if not self.mosaic:
#                 img, labels = random_perspective(img, labels,
#                                                  degrees=hyp['degrees'],
#                                                  translate=hyp['translate'],
#                                                  scale=hyp['scale'],
#                                                  shear=hyp['shear'],
#                                                  perspective=hyp['perspective'])
#
#             # Augment colorspace
#             augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
#
#             # Apply cutouts
#             # if random.random() < 0.9:
#             #     labels = cutout(img, labels)
#
#         nL = len(labels)  # number of labels
#         if nL:
#             labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
#             labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
#             labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1
#
#         if self.augment:
#             # flip up-down
#             if random.random() < hyp['flipud']:
#                 img = np.flipud(img)
#                 if nL:
#                     labels[:, 2] = 1 - labels[:, 2]
#
#             # flip left-right
#             if random.random() < hyp['fliplr']:
#                 img = np.fliplr(img)
#                 if nL:
#                     labels[:, 1] = 1 - labels[:, 1]
#
#         labels_out = torch.zeros((nL, 6))
#         if nL:
#             labels_out[:, 1:] = torch.from_numpy(labels)
#
#         # Convert
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#
#         return torch.from_numpy(img), labels_out, self.img_files[index], shapes
#
#     @staticmethod
#     def collate_fn(batch):
#         img, label, path, shapes = zip(*batch)  # transposed
#         for i, l in enumerate(label):
#             l[:, 0] = i  # add target image index for build_targets()
#         return torch.stack(img, 0), torch.cat(label, 0), path, shapes
#
# def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
#                       rank=-1, world_size=1, workers=8):
#     # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
#     with torch_distributed_zero_first(rank):
#         dataset = LoadImagesAndLabels(path, imgsz, batch_size,
#                                       augment=augment,  # augment images
#                                       hyp=hyp,  # augmentation hyperparameters
#                                       rect=rect,  # rectangular training
#                                       cache_images=cache,
#                                       single_cls=opt.single_cls,
#                                       stride=int(stride),
#                                       pad=pad,
#                                       rank=rank)
#
#     batch_size = min(batch_size, len(dataset))
#     nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
#     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batch_size,
#                                              num_workers=nw,
#                                              sampler=train_sampler,
#                                              pin_memory=True,
#                                              collate_fn=LoadImagesAndLabels.collate_fn)
#     return dataloader, dataset


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)