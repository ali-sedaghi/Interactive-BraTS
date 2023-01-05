import h5py
import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from torch.utils.data import DataLoader
from pathlib import Path
from natsort import natsorted
from isegm.data.aligned_augmentation import AlignedAugmentator
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


class BraTSDataset(ISDataset):
    def __init__(self, split, ch=0, one_input_channel=False, data_path="./data/BraTS/brats-unzipped/BraTS2020_training_data/content/data", **kwargs):
        super(BraTSDataset, self).__init__(**kwargs)
        assert split in ['train', 'val']
        self.ch = ch
        self.name = "BraTS"
        self.one_input_channel = one_input_channel
        self.data_path = Path(data_path)

        n_all = 369
        n_train = int(n_all * 75 / 100)
        n_val = n_all - n_train

        random.seed(10)
        all_indexes = list(range(1, n_all + 1))
        train_indexes = natsorted(random.sample(all_indexes, n_train))
        val_indexes = natsorted(list(set(all_indexes) - set(train_indexes)))

        if split == "train":
            indexes = train_indexes
        else:
            indexes = val_indexes

        file_names = [x.name for x in natsorted(self.data_path.glob('*.*'))]
        self.data = []

        for f_name in file_names:
            vol_num = int(f_name.split("_")[1])
            if vol_num in indexes:
                self.data.append(f_name)

        self.dataset_samples = range(len(self.data))

    def get_sample(self, index) -> DSample:
        filename = self.data_path / self.data[index]
        f = h5py.File(filename, "r")
        image_key = list(f.keys())[0]
        mask_key = list(f.keys())[1]
        image_np = f[image_key][()]     # (240, 240, 4)
        mask = f[mask_key][()]       # (240, 240, 3)
        f.close()

        img = image_np[:, :, int(self.ch)]  # (240, 240)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1) * 255
        img = img.astype('uint8')

        # wt_mask = mask[:, :, 0]
        # tc_mask = mask[:, :, 1]
        # et_mask = mask[:, :, 2]
        # tumor_mask = np.concatenate((np.expand_dims(wt_mask, axis=-1), np.expand_dims(et_mask, axis=-1)), axis=2)

        mask = np.any(mask, axis=2).astype("int32")

        if not self.one_input_channel:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return DSample(img, mask, objects_ids=[1], sample_id=index)


if __name__ == "__main__":
    ch = "0"

    train_augmentator = AlignedAugmentator(ratio=[0.3, 1.3], target_size=(96, 96), flip=True,
                                           distribution='Gaussian', gs_center=0.8, gs_sd=0.4)

    dataset = BraTSDataset('train', ch, one_input_channel=False,
                          data_path="D:\Works\Final Project\Interactive-BraTS\data\BraTS\\brats-unzipped\BraTS2020_training_data\content\data",
                           augmentator=train_augmentator)

    dataloader = DataLoader(dataset, shuffle=True)
    x = next(iter(dataloader))

    print(x['images'].shape)        # Batch of Images
    print(x['points'].shape)        # Interactive points
    print(x['instances'].shape)     # Label mask
    print(x['images'][0].shape)     # Image

    image = torch.moveaxis(x['images'][0], 0, -1)
    instance = x['instances'][0, 0]

    f, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].imshow(image)
    axs[1].imshow(instance, alpha=0.5 * instance, cmap="Reds")
    plt.show()
