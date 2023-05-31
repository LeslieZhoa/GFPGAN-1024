#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20230531
'''

import os
from torchvision import transforms 
import torchvision.transforms.functional as TF
import PIL.Image as Image
from dataloader.DataLoader import DatasetBase
import dataloader.degradations as degradations
from torchvision.transforms.functional import  normalize
import random
import math
import pickle
import numpy as np
import torch
from utils import utils
import cv2

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs

class GFPData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)
        self.eval = kwargs['eval']
        self.mean = kwargs['mean']
        self.std = kwargs['std']

      
        # degradation configurations
        self.blur_kernel_size = kwargs['blur_kernel_size']
        self.kernel_list = kwargs['kernel_list']
        self.kernel_prob = kwargs['kernel_prob']
        self.blur_sigma = kwargs['blur_sigma']
        self.downsample_range = kwargs['downsample_range']
        self.noise_range = kwargs['noise_range']
        self.jpeg_range = kwargs['jpeg_range']
        self.use_flip = kwargs['use_flip']

        self.crop_components = kwargs['crop_components']
        self.lmk_base = kwargs['lmk_base']
        self.out_size = kwargs['size']
        self.eye_enlarge_ratio = kwargs['eye_enlarge_ratio']
        
        hq_root = kwargs['hq_root']
        self.hq_paths = self.get_img_paths(hq_root)
        self.lq_root = kwargs['lq_root']
        self.img_len = 0

        if self.eval:
            
            dis1 = math.floor(len(self.hq_paths)/self.count)
            self.hq_paths = self.hq_paths[self.id*dis1:(self.id+1)*dis1]
            random.shuffle(self.hq_paths)
            

        else:
            img_root = kwargs['img_root']
            self.img_paths = self.get_img_paths(img_root)
            dis1 = math.floor(len(self.img_paths)/self.count)
            self.img_paths = self.img_paths[self.id*dis1:(self.id+1)*dis1]
            random.shuffle(self.img_paths)
            self.img_len = len(self.img_paths)

        self.hq_len = len(self.hq_paths)
        self.length = max(self.img_len,self.hq_len)
        
    def __getitem__(self,i):
        if self.eval or random.random() > 0.4:

            hq_path = self.hq_paths[i%self.hq_len]
            lq_path = os.path.join(self.lq_root,os.path.basename(hq_path))

        else:
            lq_path = hq_path = self.img_paths[i%self.img_len]

       
        
        img_hq = cv2.imread(hq_path).astype(np.float32) / 255.
        img_lq = cv2.imread(lq_path).astype(np.float32) / 255.
        # random horizontal flip
        img, status = augment([img_hq,img_lq], hflip=self.use_flip, rotation=False, return_status=True)
        img_hq,img_lq = img
        h, w, _ = img_lq.shape

        # get facial component coordinates
        loc_left_eye, loc_right_eye, loc_mouth = 0,0,0
        if self.crop_components:
            locations = self.get_component_coordinates(hq_path, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # ------------------------ generate lq image ------------------------ #
        # blur
        if not self.eval:
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
            
            img_lq = cv2.filter2D(img_lq, -1, kernel)
            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            # # noise
            if self.noise_range is not None:
                img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
            # jpeg compression
            if self.jpeg_range is not None:
                img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_hq = cv2.resize(img_hq, (1024, 1024))
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_hq, img_lq = img2tensor([img_hq, img_lq], bgr2rgb=True, float32=True)


        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        img_hq = normalize(img_hq, self.mean, self.std, inplace=True)
        img_lq = normalize(img_lq, self.mean, self.std, inplace=True)

       
        return img_lq, img_hq,loc_left_eye, loc_right_eye, loc_mouth

    def get_img_paths(self,root):
        return [os.path.join(root,f) for f in os.listdir(root)]

    
    def get_component_coordinates(self, hq_path, status):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""
        name = os.path.splitext(os.path.basename(hq_path))[0]
        info_path = os.path.join(self.lmk_base,name+'.npy')
        components_bbox = np.load(info_path,allow_pickle=True).item()
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2] / 2.
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations


    def __len__(self):
        if self.eval:
            return min(self.length,1000)
            # return 1
        else:
            return self.length
            # return 100

