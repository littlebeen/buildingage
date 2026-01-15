import torch
import glob
import random
import os
import numpy as np
from skimage import io

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, mode,main_dir,cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()
        MAIN_FOLDER = main_dir + 'hk_building_age/'+mode+'/'
        DATA_FOLDER = MAIN_FOLDER + 'image/tdop*.tif'
        LABEL_FOLDER = MAIN_FOLDER + 'class/tdop*.png'
        BOUNDARY_FOLDER = MAIN_FOLDER + 'mask/tdop*.png'
        HEIGHT_FOLDER = MAIN_FOLDER + 'height/tdop*.tif'
        self.mode = mode
        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = glob.glob(DATA_FOLDER)
        self.boundary_files = glob.glob(BOUNDARY_FOLDER)
        self.height_files = glob.glob(HEIGHT_FOLDER)
        self.label_files = glob.glob(LABEL_FOLDER)

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))


    def __len__(self):
        # Default epoch size is 10 000 samples
        return len(self.label_files)

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        data = io.imread(self.data_files[i])[:, :, :3].transpose((2, 0, 1))
        data = 1 / 255 * np.asarray(data, dtype='float32')

        height = io.imread(self.height_files[i])
        height = np.asarray(height, dtype='float32')
        height = height - height.min()
        
        boundary = np.asarray(io.imread(self.boundary_files[i])) / 255
        boundary = boundary.astype(np.int64)

        label = np.asarray(io.imread(self.label_files[i]))
        label = label.astype(np.int64)


        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        if self.mode == 'train' and self.augmentation:
            data_p, boundary_p, height_p, label_p = self.data_augmentation(data, boundary, height, label)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(boundary_p),
                torch.from_numpy(height_p),
                torch.from_numpy(label_p)-1)