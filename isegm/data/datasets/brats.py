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
    def __init__(
        self,
        data_path="",
        channel="flair",
        label="wt",
        split="train",
        **kwargs
    ):
        super(BraTSDataset, self).__init__(**kwargs)
        assert channel in ['flair', 't1', 't1ce', 't2', 'mix']
        assert label in ['net', 'ed', 'et', 'wt', 'tc']
        assert split in ['train', 'val', 'test']
        self.data_path = Path(data_path)
        self.channel = channel
        self.label = label
        self.name = "BraTS"

        n_all = 369
        n_train = int(n_all * 75 / 100)
        n_test = int(n_all * 5 / 100)

        random.seed(10)
        all_indexes = list(range(1, n_all + 1))
        train_indexes = natsorted(random.sample(all_indexes, n_train))
        val_indexes = natsorted(list(set(all_indexes) - set(train_indexes)))

        if split == "train":
            indexes = train_indexes
        elif split == "val":
            indexes = val_indexes
        elif split == "test":
            indexes = natsorted(random.sample(val_indexes, n_test))

        file_names = [x.name for x in natsorted(self.data_path.glob('*.h5'))]
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
        image_np = f[image_key][()]  # (240, 240, 4)
        mask = f[mask_key][()]  # (240, 240, 3)
        f.close()

        if self.channel == "flair":
            img = image_np[:, :, 0]
        elif self.channel == "t1":
            img = image_np[:, :, 1]
        elif self.channel == "t1ce":
            img = image_np[:, :, 2]
        elif self.channel == "t2":
            img = image_np[:, :, 3]
        elif self.channel == "mix":
            img = image_np[:, :, [0, 2, 3]]
        else:
            img = None

        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1) * 255
        img = img.astype('uint8')

        if self.channel != "mix":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        mask_net = mask[:, :, 0]
        mask_et = mask[:, :, 2]

        if self.label == "net":  # Necrosis and Non-Enhancing Tumor (NCR/NET)
            mask = mask_net
        elif self.label == "ed":  # Peritumoral Edema (ED)
            mask = mask[:, :, 1]
        elif self.label == "et":  # Enhancing Tumor (ET)
            mask = mask_et
        elif self.label == "wt":  # Whole Tumor (WT): NCR + ED + ET
            mask = np.any(mask, axis=2).astype("int32")
        elif self.label == "tc":  # Tumor Core (TC): NCR + ET
            mask = np.any(np.concatenate((
                np.expand_dims(mask_net, axis=-1),
                np.expand_dims(mask_et, axis=-1)),
                axis=2), axis=2).astype("int32")
        else:
            mask = None

        return DSample(img, mask, objects_ids=[1], sample_id=index)


if __name__ == "__main__":
    data_path = "D:\Works\Final-Project\Interactive-BraTS\data\datasets\BraTS"
    channel = "flair"  # flair, t1, t1ce, t2, mix
    label = "wt"  # net, ed, et, wt, tc

    train_augmentator = AlignedAugmentator(
        ratio=[0.3, 1.3],
        target_size=(96, 96),
        flip=True,
        distribution='Gaussian',
        gs_center=0.8,
        gs_sd=0.4
    )

    dataset = BraTSDataset(
        data_path=data_path,
        channel=channel,
        label=label,
        split='train',
        augmentator=train_augmentator
    )

    dataloader = DataLoader(dataset, shuffle=True)
    x = next(iter(dataloader))

    print(x['images'].shape)  # Batch of Images
    print(x['points'].shape)  # Interactive points
    print(x['instances'].shape)  # Label mask
    print(x['images'][0].shape)  # Image

    image = torch.moveaxis(x['images'][0], 0, -1)
    instance = x['instances'][0, 0]

    f, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].imshow(image)
    axs[1].imshow(instance, alpha=0.5 * instance, cmap="Reds")
    plt.show()
