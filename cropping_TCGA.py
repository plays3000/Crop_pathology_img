import numpy as np
import cv2
import openslide
import os
from matplotlib import pyplot as plt
from PIL import Image
import gc
import copy
from tqdm import tqdm
import pandas as pd
Image.MAX_IMAGE_PIXELS = None

class Crop_patch:
    def __init__(self, wsi_file, wsi_path, save_path, crop_size):
        self.wsi_path = wsi_path
        self.wsi_file = wsi_file
        self.save_path = save_path
        self.crop_size = crop_size
        self.level = 0

    def resize_img(self, image):
        width = int(image.shape[1] * 8)
        height = int(image.shape[0] * 8)
        dim = (width, height)
    
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized
    
    def crop_region(self, wsi_img):
        with openslide.open_slide(wsi_img) as slide:
            width, height = (slide.level_dimensions[self.level])
            if width < 2000 or height <2000:
                raise AssertionError('Width or Height is less than 10000.')
            wsi_RGB = np.array(slide.read_region((0,0), self.level, (width, height)))
            wsi_RGB = cv2.cvtColor(wsi_RGB, cv2.COLOR_BGR2RGB)
            wsi_gray = cv2.cvtColor(wsi_RGB, cv2.COLOR_RGB2GRAY)
    
        threshold = cv2.threshold(wsi_gray, 225, 255, cv2.THRESH_BINARY_INV)[1].astype('uint8')
        region = cv2.resize(wsi_RGB, (self.crop_size, self.crop_size), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(threshold, (self.crop_size, self.crop_size), interpolation = cv2.INTER_AREA)
        return region, mask
    
        # Contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # count = [len(i) for i in Contours]
        # max_contour_index = count.index(max(count))
        # max_contour = Contours[max_contour_index]
        # [X, Y, W, H] = cv2.boundingRect(max_contour)
        
        # if W < 5000 or H < 5000:
        #     raise AssertionError('Width or Height is less than 10000.')
        
        # else:
        #     c_x, c_y = int(X + (0.5*W)),int(Y + (0.5*H))
        #     if W > H:
        #         region = wsi_RGB[c_y - int(W//2):c_y + int(W//2), c_x - int(W//2):c_x + int(W//2)]
        #         mask = threshold[c_y - int(W//2):c_y + int(W//2), c_x - int(W//2):c_x + int(W//2)]
        #     else:
        #         region = wsi_RGB[c_y - int(H//2):c_y + int(H//2), c_x - int(H//2):c_x + int(H//2)]
        #         mask = threshold[c_y - int(H//2):c_y + int(H//2), c_x - int(H//2):c_x + int(H//2)]
                
        
        
    def remove_background(self, region, region_threshold):
        for i in range(region_threshold.shape[0]):
                for j in range(region_threshold.shape[1]):
                    if region_threshold[i,j] == 0:
                        region[i,j] = 0
                    else:
                        pass
        return region
    
    def patching(self, remove_bkg = True):
        region, mask = self.crop_region(self.wsi_path)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        region_removed = self.remove_background(region, mask)
        region_removed = cv2.cvtColor(region_removed, cv2.COLOR_BGR2RGB)
        plt.imsave(f'{self.save_path}/{self.wsi_file}.png', region_removed)
        
        gc.collect()
        
        
def cropping_file(crop_size, metadata_csv, save_path, test_mode = False):
    file_list = pd.read_csv(metadata_csv)['filename'].values
    wsi_list = [i for i in file_list if '.svs' in i]
    wsi_list.sort
    wsi_list = wsi_list[330:]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.system(f'sudo chown -R ubuntu:ubuntu {save_path}')
    n = 0
    
    for wsi_file in tqdm(wsi_list,  ncols = 80):
        if test_mode:
            n += 1
            wsi_path = f'./{wsi_file}'
            crop_and_patch = Crop_patch(wsi_file, wsi_path, save_path, crop_size)
            crop_and_patch.patching(remove_bkg = True)
            tqdm.write(f'{n} epoch : {wsi_file} image cropping is completed!')
        else:
            try:
                n += 1
                wsi_path = f'./{wsi_file}'
                crop_and_patch = Crop_patch(wsi_file, wsi_path, save_path, crop_size)
                crop_and_patch.patching(remove_bkg = True)
                tqdm.write(f'{n} epoch : {wsi_file} image cropping is completed!')
                
            except:
                tqdm.write(f'{n} epoch : {wsi_file} image cropping is NOT completed!')
                pass