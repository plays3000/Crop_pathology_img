import numpy as np
import cv2
import openslide
import os
from matplotlib import pyplot as plt
from PIL import Image
import gc
Image.MAX_IMAGE_PIXELS = None

class Crop_patch:
    def __init__(self,wsi_list, wsi_path, region_path, mask_path, save_path, crop_size):
        self.wsi_path = wsi_path
        self.region_path = region_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.wsi_list = wsi_list
        self.label_dir = {10 : 'WMD', 20 : 'PD' ,30 : 'PCC', 40 : 'other_CA'}
        self.crop_size = crop_size
        self.patch_size = 256

    def resize_img(self, image):
        width = int(image.shape[1] * 8)
        height = int(image.shape[0] * 8)
        dim = (width, height)
    
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized
    
    def crop_region(self):
        cropped_region = []
        cropped_mask = []
        cropped_mask_label = []

        level = 0
        region_img = cv2.imread(self.region_path, cv2.IMREAD_GRAYSCALE)
        region_img = np.invert(region_img)
        mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img2 = Image.open(self.mask_path)
        mask_img2 = np.array(mask_img2)

        contours, _ = cv2.findContours(region_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            [X, Y, W, H] = cv2.boundingRect(contour)
            new_x, new_y = int(X - (W * 0)), int(Y - (H * 0))
            new_w, new_h = int(W + (W * (0 + 0))), int(H + (H * (0 + 0)))
            mask_copy = mask_img2[new_y:new_y+new_h, new_x:new_x+new_w]
            mask_uniq = np.unique(mask_copy)
            cropped_mask_label.append(mask_uniq)
        

        ## open WSI file
        with openslide.open_slide(self.wsi_path) as slide:
            width, height = (slide.level_dimensions[level])

            wsi_BGR = np.array(slide.read_region((0, 0), level, (width, height))) #int(height/2-1200)
            wsi_RGB = cv2.cvtColor(wsi_BGR, cv2.COLOR_BGR2RGB)
        
        ## open region image and mask image and convert to grayscale to binary image
        region_binary = cv2.threshold(region_img, 0, 255, cv2.THRESH_BINARY)[1].astype('uint8')
        region_resize = self.resize_img(region_binary)
        
        mask = np.invert(mask_img)
        mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1].astype('uint8')
        mask_resize = self.resize_img(mask_binary)

        # mask_copy_resize = self.resize_img(mask_copy)
        
        ## make contour line from region bounding box
        Contours, _ = cv2.findContours(region_resize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in Contours:
            [X, Y, W, H] = cv2.boundingRect(contour)
        
            ##increase bounding box size if required
            new_x, new_y = int(X - (W * 0)), int(Y - (H * 0))
            new_w, new_h = int(W + (W * (0 + 0))), int(H + (H * (0 + 0)))
            
            cropped_region_img = wsi_RGB[new_y:new_y+new_h, new_x:new_x+new_w]
            cropped_mask_img = mask_resize[new_y:new_y+new_h, new_x:new_x+new_w]
            
            resize_cropped_region = cv2.resize(cropped_region_img, (self.crop_size, self.crop_size), interpolation = cv2.INTER_AREA)
            resize_cropped_mask = cv2.resize(cropped_mask_img, (self.crop_size, self.crop_size), interpolation = cv2.INTER_AREA)

            cropped_region.append(resize_cropped_region)
            cropped_mask.append(resize_cropped_mask)
            
        return (cropped_region, cropped_mask, cropped_mask_label)
    
    def remove_background(self, region, region_threshold):
        for i in range(region_threshold.shape[0]):
                for j in range(region_threshold.shape[1]):
                    if region_threshold[i,j] == 0:
                        region[i,j] = 0
                    else:
                        pass
        return region
    
    def patching(self, remove_bkg = False):
        cropped_region, _, cropped_mask_label = self.crop_region()
        k = 0
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        for region, mask in zip(cropped_region, cropped_mask_label):
            k += 1

            if len(mask)>=3:
                label = 'mixed'
            else:
                label = self.label_dir[mask[1]]

            region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            region_threshold = cv2.threshold(region_gray, 230,255, cv2.THRESH_BINARY)[1]
            region_threshold = np.invert(region_threshold)

            if remove_bkg is True:
                region_removed = self.remove_background(region, region_threshold)
                
                for n in range(int(region_removed.shape[0]//self.patch_size)):
                    for m in range(int(region_removed.shape[1]//self.patch_size)):
                        mask_tile = region_threshold[(self.patch_size*n):self.patch_size+(self.patch_size*n), (self.patch_size*m):self.patch_size+(self.patch_size*m)]
                        region_removed_tile = region_removed[(self.patch_size*n):self.patch_size+(self.patch_size*n), (self.patch_size*m):self.patch_size+(self.patch_size*m)]
                        if np.mean(mask_tile)<=100:
                            pass
                        else:
                            if not os.path.exists(f'{self.save_path}/{label}'):
                                os.mkdir(f'{self.save_path}/{label}')
                                os.system(f'sudo chown -R ubuntu:ubuntu {self.save_path}/{label}')
                            plt.imsave(f'{self.save_path}/{label}/{self.wsi_list}_{k}_{n}_{m}.png', region_removed_tile)
                            
            else:
                for n in range(int(region.shape[0]//self.patch_size)):
                    for m in range(int(region.shape[1]//self.patch_size)):
                        mask_tile = region_threshold[(self.patch_size*n):self.patch_size+(self.patch_size*n), (self.patch_size*m):self.patch_size+(self.patch_size*m)]
                        region_tile = region[(self.patch_size*n):self.patch_size+(self.patch_size*n), (self.patch_size*m):self.patch_size+(self.patch_size*m)]
                        
                        if np.mean(mask_tile)<=100:
                            pass
                        else:
                            if not os.path.exists(f'{self.save_path}/{label}'):
                                os.mkdir(f'{self.save_path}/{label}')
                                os.system(f'sudo chown -R ubuntu:ubuntu {self.save_path}/{label}')
                            plt.imsave(f'{self.save_path}/{label}/{self.wsi_list}_{k}_{n}_{m}.png', region_tile)
        
        gc.collect()
        
        
def cropping_file(args):
    hos, crop_size, save_path = args
    wsi_list = os.listdir(f'./{hos}/Slides')
    region_list = os.listdir(f'./{hos}/{hos}_region')
    mask_list = os.listdir(f'./{hos}/{hos}_total')
    wsi_list.sort()
    region_list.sort()
    mask_list.sort()

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.system(f'sudo chown -R ubuntu:ubuntu {save_path}')
    n = 0
    for wsi_file, region_file, mask_file in zip(wsi_list, region_list, mask_list):
        n += 1
        mask_path = f'./{hos}/{hos}_total/'+ mask_file
        wsi_path = f'./{hos}/Slides/'+wsi_file
        region_path = f'./{hos}/{hos}_region/'+region_file
        try:
            crop_and_patch = Crop_patch(wsi_file, wsi_path, region_path, mask_path, save_path, crop_size)
            crop_and_patch.patching(remove_bkg = False)
            print(f'{n} epoch : {wsi_file} image cropping is completed!')
        except:
            print(f'{n} epoch : {wsi_file} image cropping is NOT completed!')
            pass
                  
if __name__ == '__main__':
    hos = 'sevrance'
    crop_size = 4096
    save_path = './test_folder'
    args = (hos, crop_size, save_path)
    cropping_file(args)

 #==================================================================================================
    # Add this code into patching function if you need to seperate normal cropped image and abnormal cropped image.

        # abnormal_region = np.copy(region)
        # normal_region = np.copy(region)
        # abnormal_mask = np.copy(region_threshold)
        # normal_mask = np.copy(region_threshold)

        #make abnormal region mask
        # for i in range(region_threshold.shape[0]):
        #     for j in range(region_threshold.shape[1]):
        #         if mask[i,j] == 0:
        #             abnormal_mask[i,j] = 0
        #         else:
        #             pass
        # # make abnormal_region
        # for i in range(abnormal_region.shape[0]):
        #     for j in range(abnormal_region.shape[1]):
        #         if region_threshold[i,j] == 0:
        #             abnormal_region[i,j,:] = [0,0,0]
        #         else:
        #             pass
        
        # #make normal region mask
        # for i in range(normal_mask.shape[0]):
        #     for j in range(normal_mask.shape[1]):
        #         if abnormal_mask[i,j] == 0:
        #             normal_mask[i,j] = 0
        #         else:
        #             pass

        # #make normal region
        # for i in range(normal_region.shape[0]):
        #     for j in range(normal_region.shape[1]):
        #         if normal_mask[i,j] == 0:
        #             normal_region[i,j,:] = [0,0,0]
        #         else:
        #             pass
         #make normal patch and save the images
        # for n in range(int(normal_region.shape[0]//256)):
        #     for m in range(int(normal_region.shape[1]//256)):
        #         mask_tile = normal_mask[(224*n):224+(224*n), (224*m):224+(224*m)]
        #         nor_tile = normal_region[(224*n):224+(224*n), (224*m):224+(224*m)]
        #         if np.mean(mask_tile)<100:
        #             pass
        #         else:
        #             if not os.path.exists(f'{self.save_path}/normal'):
        #                 os.mkdir(f'{self.save_path}/normal')
        #             plt.imsave(f'{self.save_path}/normal/{self.wsi_list}_{n}_{m}.png',nor_tile)

        #make abnormal patch and save the images
        # for n in range(int(abnormal_region.shape[0]//224)):
        #     for m in range(int(abnormal_region.shape[1]//224)):
        #         mask_tile = abnormal_mask[(224*n):224+(224*n), (224*m):224+(224*m)]
        #         abn_tile = abnormal_region[(224*n):224+(224*n), (224*m):224+(224*m)]
        #         if np.mean(mask_tile)<=100:
        #             pass
        #         else:
        #             if not os.path.exists(f'{self.save_path}/{label}'):
        #                 os.mkdir(f'{self.save_path}/{label}')
        #             plt.imsave(f'{self.save_path}/{label}/{self.wsi_list}_{n}_{m}.png', abn_tile)
#======================================================================================================