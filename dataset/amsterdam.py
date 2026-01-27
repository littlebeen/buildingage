import torch
import glob
import random
import os
import numpy as np
from skimage import io

def get_year_type(arr_processed):
 
    arr_processed[(arr_processed < 1980) & (arr_processed > 1)] = 1
    arr_processed[(arr_processed >= 1980) & (arr_processed <= 2000)] = 2
    arr_processed[arr_processed > 2000] = 3
    return arr_processed
class Amsterdam_dataset(torch.utils.data.Dataset):
    def __init__(self, mode,cache=False, augmentation=True):
        super(Amsterdam_dataset, self).__init__()
        MAIN_FOLDER = '../dataset/Amsterdam/'+mode+'/'
        DATA_FOLDER = MAIN_FOLDER + 'image/*.tiff'
        LABEL_FOLDER = MAIN_FOLDER + 'age/*.tiff' 
        self.mode = mode
        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = glob.glob(DATA_FOLDER)
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

        label = np.asarray(io.imread(self.label_files[i]))
        label = label.astype(np.int64)
        label = get_year_type(label)
        # unique_values1 = np.unique(label)
        # print(unique_values1)
        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        if self.mode == 'train' and self.augmentation:
            data, label = self.data_augmentation(data, label)

        # Return the torch.Tensor values
        return (torch.from_numpy(data),
                torch.from_numpy(data),
                torch.from_numpy(data),
                torch.from_numpy(label)-1)