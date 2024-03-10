import os
import natsort
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from DRAEM_module.perlin import rand_perlin_2d_np
from scipy.stats import norm

#### Test ############################################################################################################

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, args):
        self.root_dir = os.path.join(args['save_dir'], "raw_data")

        img_name_list = natsort.natsorted(os.listdir(self.root_dir))
        self.images = []
        for img_name in img_name_list:
            self.images.append(os.path.join(self.root_dir, img_name))

        self.resize_shape = args['target_size']
        self.preprocessing = args['Preprocessing_methods']
        #self.anomaly_dir = args['anomaly_source_path']

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        #### Preprocessing ####################################
    
        def normalization(img):
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return img

        def histogram_equalization(img):
            img = cv2.equalizeHist(img)
            return img

        def normalization_and_histogram_equalization(img):
            img = normalization(img)
            img = img.astype(np.uint8)
            img = histogram_equalization(img)
            return img
    
        if self.preprocessing == "Normalization":
            image = normalization(image)
        
        elif self.preprocessing == "Histogram Equalization":
            image = histogram_equalization(image)
        
        elif self.preprocessing =="Normalization + Histogram Equalization":
            image = normalization_and_histogram_equalization(image)

        #######################################################
        # ex) normalize, hist_equalize
        # 적용 확인 완료 

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[0], self.resize_shape[1]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[0], self.resize_shape[1]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 1)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'ACC':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx, 'file_name' : file_name}

        return sample




#### Train ############################################################################################################

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, args):

        self.preprocessing = args['Preprocessing_methods']
        self.root_dir = args['data_path']
        self.resize_shape = args['target_size']


        img_name_list = natsort.natsorted(os.listdir(self.root_dir))
        self.image_paths = []
        for img_name in img_name_list:
            self.image_paths.append(os.path.join(self.root_dir, img_name))

        anormaly_path = "/2023_BTS/dtd/images/"
        self.anomaly_source_paths = sorted(glob.glob(anormaly_path +  "*/*.jpg"))
        

        self.augmenters = [
                      iaa.GammaContrast((0.5,2.0),per_channel=False),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.root_dir)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path, cv2.IMREAD_GRAYSCALE)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_source_img = np.expand_dims(anomaly_source_img, axis=2)

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        #### Preprocessing ####################################
    
        def normalization(img):
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return img

        def histogram_equalization(img):
            img = cv2.equalizeHist(img)
            return img

        def normalization_and_histogram_equalization(img):
            img = normalization(img)
            img = img.astype(np.uint8)
            img = histogram_equalization(img)
            return img
    
        if self.preprocessing == "Normalization":
            image = normalization(image)
        
        elif self.preprocessing == "Histogram Equalization":
            image = histogram_equalization(image)
        
        elif self.preprocessing =="Normalization + Histogram Equalization":
            image = normalization_and_histogram_equalization(image)

        #######################################################
        # ex) normalize, hist_equalize
        # 적용 확인 완료 


        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.expand_dims(image, axis=2)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0      
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
    
        return sample
    




#### Inference ############################################################################################################

class MVTecDRAEMInferenceDataset(Dataset):

    def __init__(self, args):
        self.root_dir = os.path.join(args['save_dir'], "raw_data")

        img_name_list = natsort.natsorted(os.listdir(self.root_dir))
        self.images = []
        for img_name in img_name_list:
            self.images.append(os.path.join(self.root_dir, img_name))
        self.resize_shape = args['target_size']
        self.preprocessing = args['Preprocessing_methods']

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        #### Preprocessing ####################################
    
        def normalization(img):
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return img

        def histogram_equalization(img):
            img = cv2.equalizeHist(img)
            return img

        def normalization_and_histogram_equalization(img):
            img = normalization(img)
            img = img.astype(np.uint8)
            img = histogram_equalization(img)
            return img
    
        if self.preprocessing == "Normalization":
            image = normalization(image)
        
        elif self.preprocessing == "Histogram Equalization":
            image = histogram_equalization(image)
        
        elif self.preprocessing =="Normalization + Histogram Equalization":
            image = normalization_and_histogram_equalization(image)

        #######################################################


        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 1)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'ACC':
            image, mask = self.transform_image(img_path, None)
            #has_anomaly = np.array([0], dtype=np.float32)
        else:
            image, mask = self.transform_image(img_path, None)
            #has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'mask': mask, 'idx': idx, 'file_name' : file_name}

        return sample