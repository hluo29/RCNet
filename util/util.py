import glob
import sys, time, os

import shutil, logging
from datetime import datetime

import random, cv2, collections
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from skimage.measure import compare_psnr, compare_ssim


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def setup_logger(logger_name, log_dir, phase, level=logging.INFO, screen=False, tofile=False):
    # initialize a logger
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    time_stamp = datetime.now().strftime('%y%m%d-%H%M%S')
    
    if tofile:
        # create file_handler for logger
        log_file = os.path.join(log_dir, phase+'_{}_{}.log'.format(logger_name, time_stamp))
        fh = logging.FileHandler(log_file, mode='w')        
        fh.setFormatter(formatter)
        log.addHandler(fh)
    if screen:
        # create console_handler for logger
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), 'The path still has not been made !!!' % path
    return path

def process_bar(num, total, index):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+' dataset_id:{} [{}{}] {}%\n'.format(index, '*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()

def boundaryCheck(imgpaths, use_status=True):
    for imgpath in imgpaths:
        if is_image_file(imgpath):
            img = cv2.imread(imgpath)
            h, w = img.shape[0], img.shape[1]
            zero_count_up = dict(collections.Counter(img[0, :, 0])).get(0, 0)
            zero_count_down = dict(collections.Counter(img[h - 1, :, 0])).get(0, 0)
            zero_count_left = dict(collections.Counter(img[:, 0, 0])).get(0, 0)
            zero_count_right = dict(collections.Counter(img[:, w - 1, 0])).get(0, 0)
            if np.all(img[0, :, :]==0) or np.all(img[h-1, :, :]==0) or np.all(img[:, 0, :]==0) or np.all(img[:, w-1, :]==0) \
                    or zero_count_up>=320 or zero_count_down>=320 or zero_count_left>=320 or zero_count_right>=320:
                use_status = False
                break
        else:
            use_status = False
    return use_status

class RandomHorizontallyFlip(object):
    """CV2 library, shape-(h,w,c)"""
    def __call__(self, img_1, img_2, img_3, gt_2):
        if random.random() < 0.5:   # random.random()-->[0,1), [0,50) and [50,100)
            return img_1[:,::-1,:], img_2[:,::-1,:], img_3[:,::-1,:], gt_2[:,::-1,:]
        else:
            return img_1, img_2, img_3, gt_2

class RandomVertivallyFlip(object):
    """CV2 library, shape-(h,w,c)"""
    def __call__(self, image, gt):
        if random.random() < 0.5:   # random.random()-->[0,1), [0,50) and [50,100)
            return image[::-1,:,:], gt[::-1,:,:]
        else:
            return image, gt

def tensor2img(input_image, need_reciprocal=False, is_pretrained=False, imtype=np.uint8):
    """Convert a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) -- tensor array
        imtype (type)        -- desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): # get the data from a variable
            image_tensor = input_image
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().data.numpy() # convert into a numpy array
        if is_pretrained:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            print('re-nomalize images for saving')
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        if need_reciprocal:
            image_numpy = 1./image_numpy
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype) # convert into a desired type

def save_image(image_numpy, image_path):
    """Convert a numpy image into a Pillow image and save to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

