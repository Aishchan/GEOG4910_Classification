import os

import numpy as np
import rasterio
import torch


# Dataset class
class TM_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        super(TM_Dataset, self).__init__()
        
        # define data path
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.label_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         self.transform = transform
    
    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        
        rsimage = rasterio.open(self.image_paths[idx]) # open tif by rasterio
        image = rsimage.read([1,2,3,4,5,6,7]) # load bands
        image = 1/255 * np.asarray(image, dtype='float32') # normalize images
        
        rslabel = rasterio.open(self.label_paths[idx]) # open tif by rasterio
        label = rslabel.read(1) # load label image
#         label = np.expand_dims(label, axis=0)

        # convert label from 0-3(4 classes)
        label[label==1] = 0
        label[label==2] = 1
        label[label==3] = 2
        label[label==4] = 3
        
        # Data augmentation
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        
        rsimage.close()
        rslabel.close()
                           
        return image_tensor, label_tensor


class Sentinel_2_Dataset_8bit(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        super(Sentinel_2_Dataset_8bit, self).__init__()
        
        # define data path
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.label_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         self.transform = transform
    
    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        rsimage = rasterio.open(self.image_paths[idx]) # open tif by rasterio
        image = rsimage.read([1,2,3,4,5,6,7]) # 
        image = 1/255 * np.asarray(image, dtype='float32') # normalize images
        
        rslabel = rasterio.open(self.label_paths[idx]) # open tif by rasterio
        label = rslabel.read(1) # load label image
#         label = np.expand_dims(label, axis=0)

        # convert label from 0-3(4 classes)
        label[label==1] = 1
        label[label==2] = 2
        label[label==3] = 3
        label[label==4] = 4
        label[label==5] = 5
        
        # Data augmentation
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        
        rsimage.close()
        rslabel.close()
                           
        return image_tensor, label_tensor

class Sentinel_2_Dataset_16bit(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        super(Sentinel_2_Dataset_16bit, self).__init__()
        
        # define data path
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.label_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         self.transform = transform
    
    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        rsimage = rasterio.open(self.image_paths[idx]) # open tif by rasterio
        image = rsimage.read([1,2,3,4,5,6,7]) # 
        image = 1/255 * np.asarray(image, dtype='float32') # normalize images
        
        rslabel = rasterio.open(self.label_paths[idx]) # open tif by rasterio
        label = rslabel.read(1) # load label image
#         label = np.expand_dims(label, axis=0)

        # convert label from 0-3(4 classes)
        label[label==1] = 1
        label[label==2] = 2
        label[label==3] = 3
        label[label==4] = 4
        label[label==5] = 5
        
        # Data augmentation
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        
        rsimage.close()
        rslabel.close()
                           
        return image_tensor, label_tensor

class Landsat_Dataset_8bit(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        super(Landsat_Dataset_8bit, self).__init__()
        
        # define data path
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.label_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         self.transform = transform
    
    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        rsimage = rasterio.open(self.image_paths[idx]) # open tif by rasterio
        image = rsimage.read([1,2,3,4,5,6,7]) # 
        image = 1/255 * np.asarray(image, dtype='float32') # normalize images
        
        rslabel = rasterio.open(self.label_paths[idx]) # open tif by rasterio
        label = rslabel.read(1) # load label image
#         label = np.expand_dims(label, axis=0)

        # convert label from 0-3(4 classes)
        label[label==1] = 1
        label[label==2] = 2
        label[label==3] = 3
        label[label==4] = 4
        label[label==5] = 5
        
        # Data augmentation
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        
        rsimage.close()
        rslabel.close()
                           
        return image_tensor, label_tensor

class Landsat_Dataset_16bit(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        super(Landsat_Dataset_16bit, self).__init__()
        
        # define data path
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.label_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         self.transform = transform
    
    def __len__(self):
        
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        rsimage = rasterio.open(self.image_paths[idx]) # open tif by rasterio
        image = rsimage.read([1,2,3,4,5,6,7]) # 
        image = 1/255 * np.asarray(image, dtype='float32') # normalize images
        
        rslabel = rasterio.open(self.label_paths[idx]) # open tif by rasterio
        label = rslabel.read(1) # load label image
#         label = np.expand_dims(label, axis=0)

        # convert label from 0-3(4 classes)
        label[label==1] = 1
        label[label==2] = 2
        label[label==3] = 3
        label[label==4] = 4
        label[label==5] = 5
        
        # Data augmentation
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        
        rsimage.close()
        rslabel.close()
                           
        return image_tensor, label_tensor
