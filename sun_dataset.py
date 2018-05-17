import os
from PIL import Image
from torch.utils.data import Dataset
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


def get_data_loader(opt):
    transform = transforms.Compose([
        transforms.Resize([opt.reso, opt.reso]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    if opt.is_train:
        train_set = SunDataset(
            root_dir='assets',
            train=True,
            transform=transform,
        )
        return DataLoader(
            train_set,
            batch_size=opt.bs,
            shuffle=True,
            num_workers=opt.num_preprocess_workers,
        )
    else:
        test_set = SunDataset(
            root_dir='assets',
            train=False,
            transform=transform,
        )
        return DataLoader(
            test_set,
            batch_size=opt.bs,
            shuffle=False,
            num_workers=opt.num_preprocess_workers,
        )


class SunDataset(Dataset):
    def __init__(self, root_dir=None, train=None, transform=None, target_height=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.target_height = target_height
        if transform and target_height is None:
            raise Exception("If transform has given, target height must be specified.")
        if self.train:
            with open(os.path.join(self.root_dir, 'final_annotations.pkl'), 'rb') as f:
                self.index_table = pickle.load(f)
        else:
            raise Exception("Test data not defined.")

        self.label_length = 5841
        self.max_boxes = 149

    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, idx):
        sample = dict()
        image_dir = os.path.join(self.index_table[idx]['directory'], self.index_table[idx]['image_path'])
        sample['img'] = Image.open(image_dir)
        raw_label = self.index_table[idx]['boxes']

        if self.transform is not None:
            width, height = sample['img'].size
            sample['img'] = self.transform(sample['img'])
            label = np.zeros([self.max_boxes, self.label_length + 5]).astype(np.float32)
            for box_index in range(len(sample['label'])):
                x_max = int(raw_label[box_index]['x_max'] / width * self.target_height)
                x_min = int(raw_label[box_index]['x_min'] / width * self.target_height)
                y_max = int(raw_label[box_index]['y_max'] / height * self.target_height)
                y_min = int(raw_label[box_index]['y_min'] / height * self.target_height)
                label[box_index][0] = float(x_min)
                label[box_index][1] = float(y_min)
                label[box_index][2] = float(x_max)
                label[box_index][3] = float(y_max)
                label[box_index][4] = 1.0
                label[box_index][raw_label[box_index]['label'] + 5] = 1.0
            sample['label'] = label
        return sample
