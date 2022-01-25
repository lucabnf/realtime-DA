import torch
import os
from torchvision import transforms
import glob
from utils import get_IDDA_label_info, get_CamVid_label_info, RandomCrop, one_hot_it_v11_dice, one_hot_it_v11,\
    augmentation, augmentation_pixel
from PIL import Image
import numpy as np
import random


class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, json_path, csv_path, scale, loss='dice'):
        super().__init__()
        # List of images
        self.image_list = []
        self.image_list.extend(glob.glob(os.path.join(image_path, '*.jpg')))
        self.image_list.sort()

        # List of labels
        self.label_list = []
        self.label_list.extend(glob.glob(os.path.join(label_path, '*.png')))
        self.label_list.sort()

        # Mapping dictionary IDDA - CamVid classes
        self.toCamVidDict = {0: [0, 128, 192],
                             1: [128, 0, 0],
                             2: [64, 0, 128],
                             3: [192, 192, 128],
                             4: [64, 64, 128],
                             5: [64, 64, 0],
                             6: [128, 64, 128],
                             7: [0, 0, 192],
                             8: [192, 128, 128],
                             9: [128, 128, 128],
                             10: [128, 128, 0],
                             255: [0, 0, 0]}

        # Info about labels
        self.camvid_label_info = get_CamVid_label_info(csv_path)
        self.idda_labels_info = get_IDDA_label_info(json_path)

        # Normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # open image and label
        img = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index]).convert("RGB")

        # resize image and label, then crop them
        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        seed = random.random()
        img = transforms.Resize(scale, Image.BILINEAR)(img)
        img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        img = np.array(img)

        label = transforms.Resize(scale, Image.NEAREST)(label)
        label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        label = np.array(label)

        # re-assign the labels to match the format of CamVid
        new_label = np.zeros(label.shape, dtype=np.uint8)
        for i in range(len(self.idda_labels_info['label2camvid'])):
            # True if IDDA label is in the labels that will be mapped with CamVid labels
            mask = label[:, :, 0] == self.idda_labels_info['label2camvid'][i][0]
            # Translate the IDDA labels with the corresponding CamVid labels 
            new_label[mask] = self.toCamVidDict[self.idda_labels_info['label2camvid'][i][1]]

        label = new_label

        # apply augmentation (both on whole image and on pixels)
        if random.randint(0,1) == 1:
            img, label = augmentation(img, label)

        if random.randint(0, 1) == 1:
            img = augmentation_pixel(img)

        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        # computing losses
        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.camvid_label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.camvid_label_info).astype(np.uint8)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)