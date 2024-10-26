from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from os import listdir
from os.path import isfile, join

import torch
import torch.utils.data as data

from scripts.utils.osutils import *
from scripts.utils.imutils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COMIC(data.Dataset):
    def __init__(self, train, config=None, sample=[], gan_norm=False, use_wm=True):
        self.train = []
        self.anno = []
        self.mask = []
        self.wm = [] if use_wm else None
        self.input_size = config.input_size
        self.normalized_input = config.normalized_input
        self.base_folder = config.base_dir
        self.dataset = train + config.data

        if config is None:
            self.data_augumentation = False
        else:
            self.data_augumentation = config.data_augumentation

        self.istrain = False if self.dataset.find("train") == -1 else True
        self.sample = sample
        self.gan_norm = gan_norm
        mypath = join(self.base_folder, self.dataset)
        file_names = sorted(
            [
                f
                for f in listdir(join(mypath, "image"))
                if isfile(join(mypath, "image", f))
            ]
        )

        if config.limited_dataset > 0:
            xtrain = sorted(
                list(set([file_name.split("-")[0] for file_name in file_names]))
            )
            tmp = []
            for x in xtrain:
                # get the file_name by identifier
                tmp.append([y for y in file_names if x in y][0])

            file_names = tmp
        else:
            file_names = file_names

        for file_name in file_names:
            self.train.append(os.path.join(mypath, "image", file_name))
            self.mask.append(os.path.join(mypath, "mask", file_name))
            if use_wm:
                self.wm.append(os.path.join(mypath,'watermark',file_name))
            self.anno.append(
                os.path.join(
                    # self.base_folder, 
                    mypath,
                    "target", 
                    # file_name,
                    file_name.split("-")[0] + ".png"
                )
            )

        if len(self.sample) > 0:
            self.train = [self.train[i] for i in self.sample]
            self.mask = [self.mask[i] for i in self.sample]
            self.anno = [self.anno[i] for i in self.sample]

        self.trans = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )

        print("total Dataset of " + self.dataset + " is : ", len(self.train))

    def __getitem__(self, index):
        img = Image.open(self.train[index]).convert("RGB")
        mask = Image.open(self.mask[index]).convert("L")
        # NOTE: Hotfix WR_ser_manhua_ds3
        try:
            anno = Image.open(self.anno[index]).convert("RGB")
        except Exception:
            anno = Image.open(self.mask[index].replace("mask", "target")).convert("RGB")
        wm = None
        if self.wm is not None:
            wm = Image.open(self.wm[index]).convert('RGB')

        # # if mask has no masked region, return random sample
        # if np.sum(np.array(mask)) == 0:
        #     return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "image": self.trans(img),
            "target": self.trans(anno),
            "mask": self.trans(mask),
            "wm": self.trans(wm) if wm is not None else None,
            "name": self.train[index].split("/")[-1],
            "imgurl": self.train[index],
            "maskurl": self.mask[index],
            "targeturl": self.anno[index],
            "wmurl": self.wm[index] if wm is not None else None,
        }

    def __len__(self):
        return len(self.train)
        # return 10


class COMBINED_COMMIC(data.Dataset):
    def __init__(self, train, config=None, sample=[], gan_norm=False):
        self.primary_ds = COMIC(f"{train}/primary", config, sample, gan_norm)
        self.secondary_ds = COMIC(f"{train}/secondary", config, sample, gan_norm)

        self.primary_len = len(self.primary_ds)
        self.secondary_len = len(self.secondary_ds)

        assert self.primary_len < self.secondary_len

    def __getitem__(self, index):
        if index < self.primary_len:
            return self.primary_ds[index]
        else:
            index = random.randint(0, self.secondary_len - 1)
            return self.secondary_ds[index]

    def __len__(self):
        return 2 * min(self.primary_len, self.secondary_len)


if __name__ == "__main__":
    from tqdm import tqdm
    
    class Config:
        input_size = 256
        normalized_input = None
        base_dir = "../tmp/split_and_refine_ds"
        data = ""
        data_augumentation = False
        limited_dataset = 0

    ds = COMIC("val", Config)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    for batch in tqdm(dl):
        # print(batch["image"].shape)
        pass

    ds = COMIC("train", Config)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    for batch in tqdm(dl):
        # print(batch["image"].shape)
        pass
