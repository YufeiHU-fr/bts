# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random
import cv2
import sys
import json
import PIL
import warnings

from distributed_sampler_no_evenly_divisible import *

def cutout(img, i, j, h, w, v, inplace=False):
    """ Erase the CV Image with given value.
    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.
    Returns:
        CV Image: Cutout image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.copy()

    img[i:i + h, j:j + w, :] = v
    return img

class Cutout(object):
    """Random erase the given CV Image.
    It has been proposed in
    `Improved Regularization of Convolutional Neural Networks with Cutout`.
    `https://arxiv.org/pdf/1708.04552.pdf`
    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        pixel_level (bool): filling one number or not. Default value is False
    """
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.pixel_level = pixel_level
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio):
        if type(img) == np.ndarray:
            img_h, img_w, img_c = img.shape
        else:
            img_h, img_w = img.size
            img_c = len(img.getbands())

        s = random.uniform(*scale)
        # if you img_h != img_w you may need this.
        # r_1 = max(r_1, (img_h*s)/img_w)
        # r_2 = min(r_2, img_h / (img_w*s))
        r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if random.random() < self.p:
            left, top, h, w, ch = self.get_params(img, self.scale, self.ratio)

            if self.pixel_level:
                c = np.random.randint(*self.value, size=(h, w, ch), dtype='uint8')
            else:
                c = random.randint(*self.value)

            if type(img) == np.ndarray:
                return cutout(img, top, left, h, w, c, self.inplace)
            else:
                if self.pixel_level:
                    c = PIL.Image.fromarray(c)
                img.paste(c, (left, top, left + w, top + h))
                return img
        return img

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode,cutout_=True, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.cutout=cutout_
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        if self.args.dataset == 'cityscapes':
            camera_path = sample_path.split()[2]
            json_camera = open(camera_path)
            data_camera = json.load(json_camera)
            baseline = data_camera['extrinsic']['baseline']
            focal = (data_camera['intrinsic']['fx']+data_camera['intrinsic']['fy'])/2.
            json_camera.close()
        else:
            focal = float(sample_path.split()[2])

        if self.mode == 'train':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[3])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[4])
            elif self.args.dataset=='cityscapes':
                image_path = sample_path.split()[0]
                depth_path = sample_path.split()[1]
            else:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[1])

            if self.args.dataset=='cityscapes':
                image = Image.open(image_path)
                depth_gt = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED).astype(np.float16)
                depth_gt[depth_gt > 0]= (depth_gt[depth_gt > 0] - 1) / 256
                gt_height, gt_width = depth_gt.shape
                depth_gt = (float(baseline) * float(focal)) / depth_gt
                depth_gt[depth_gt == np.inf] = 0
                depth_gt = depth_gt.astype(np.uint8)
                depth_gt = Image.fromarray(depth_gt)
                self.args.do_kb_crop=False
            else:
                image = Image.open(image_path)
                depth_gt = Image.open(depth_path)
            
            if self.args.do_kb_crop is True:
                print('train crop')
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))
    
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            elif self.args.dataset == 'cityscapes':
                depth_gt = depth_gt
            else:
                depth_gt = depth_gt / 256.0
            
            #print('original:',image.shape)1024,2048
            image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            #print('resize:',image.shape)512,1024
            #exit()
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path
            if self.args.dataset == 'cityscapes':
                image_path = sample_path.split()[0]
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
            else:
                image_path = os.path.join(data_path, "./" + sample_path.split()[0])
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                if self.args.dataset == 'cityscapes':
                    depth_path = sample_path.split()[1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float16)
                    depth_gt[depth_gt > 0] = (depth_gt[depth_gt > 0] - 1) / 256
                    gt_height, gt_width = depth_gt.shape
                    depth_gt = (float(baseline) * float(focal)) / depth_gt
                    depth_gt[depth_gt == np.inf] = 0
                    depth_gt = depth_gt.astype(np.uint8)
                    depth_gt = Image.fromarray(depth_gt)
                    has_valid_depth = True
                    self.args.do_kb_crop=False
                else:
                    gt_path = self.args.gt_path_eval
                    depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                    has_valid_depth = False
                    try:
                        depth_gt = Image.open(depth_path)
                        has_valid_depth = True
                    except IOError:
                        depth_gt = False
                        # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    elif self.args.dataset=='cityscapes':
                        depth_gt=depth_gt
                    else:
                        depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                print('crop')
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            
            if self.mode == 'online_eval':
                has_valid_depth=True
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
       
        #print(sample)
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        if self.cutout:
            img_h, img_w, img_c = image.shape
            left = random.randint(0, img_w - 64)
            top = random.randint(0, img_h - 64)
            image = cutout(image, top, left, 64, 64, 0)
            depth_gt = cutout(depth_gt, top, left, 64, 64, 0)
            #print('cutout!')
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


if __name__=="__main__":
    import argparse


    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--model_name', type=str, help='model name', default='bts_eigen_v2')
    parser.add_argument('--encoder', type=str, help='type of encoder, desenet121_bts, densenet161_bts, '
                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                        default='densenet161_bts')
    # Dataset
    parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--data_path', type=str, help='path to the data', required=True)
    parser.add_argument('--gt_path', type=str, help='path to the groundtruth data', required=True)
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    # Log and save
    parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
    parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=100)
    parser.add_argument('--save_freq', type=int, help='Checkpoint saving frequency in global steps', default=500)

    # Training
    parser.add_argument('--fix_first_conv_blocks', help='if set, will fix the first two conv blocks',
                        action='store_true')
    parser.add_argument('--fix_first_conv_block', help='if set, will fix the first conv block', action='store_true')
    parser.add_argument('--bn_no_track_stats', help='if set, will not track running stats in batch norm layers',
                        action='store_true')
    parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--bts_size', type=int, help='initial num_filters in bts', default=512)
    parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                        action='store_true')
    parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-6)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate', type=float, help='end learning rate', default=-1)
    parser.add_argument('--variance_focus', type=float,
                        help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
                        default=0.85)

    # Preprocessing
    parser.add_argument('--do_random_rotate', help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    # Multi-gpu training
    parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
    parser.add_argument('--world_size', type=int, help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training',
                        default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')
    parser.add_argument('--gpu', type=int, help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed', help='Use multi-processing distributed training to launch '
                                                              'N processes per node, which has N GPUs. This is the '
                                                              'fastest way to use PyTorch for either single node or '
                                                              'multi node data parallel training',
                        action='store_true', )
    # Online eval
    parser.add_argument('--do_online_eval', help='if set, perform online eval in every eval_freq steps',
                        action='store_true')
    parser.add_argument('--data_path_eval', type=str, help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval', type=str, help='path to the groundtruth data for online evaluation',
                        required=False)
    parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file for online evaluation',
                        required=False)
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--eval_freq', type=int, help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--eval_summary_directory', type=str, help='output directory for eval summary,'
                                                                   'if empty outputs to checkpoint folder', default='')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    if args.mode == 'train' and not args.checkpoint_path:
        from bts import *

    elif args.mode == 'train' and args.checkpoint_path:
        model_dir = os.path.dirname(args.checkpoint_path)
        model_name = os.path.basename(model_dir)
        import sys

        sys.path.append(model_dir)
        for key, val in vars(__import__(model_name)).items():
            if key.startswith('__') and key.endswith('__'):
                continue
            vars()[key] = val

    args.distributed = False
    dataloader =  BtsDataLoader(args, 'online_eval')
    print(len(dataloader.data))
    import utils
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for i, input in enumerate(dataloader.data):
        result = input['depth'].squeeze().detach().cpu().numpy()
        #print(result.shape)
        #exit()
        result = (denorm(result) * 255).transpose(1, 2, 0).astype(np.uint8)
        result = np.clip(result,0,255)
        Image.fromarray(result).save(os.path.join('./visualization', '{}-img_depth.jpg'.format(i)))
        print(i+1)
        print(input['image'].shape)
        print(input['focal'])
        exit()
    #import tqdm
    #for _, eval_sample_batched in enumerate(tqdm(dataloader.data)):
    #    print(eval_sample_batched)
