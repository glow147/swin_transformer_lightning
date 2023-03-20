import torch
import torchvision
import xml.etree.ElementTree as ET

from pathlib import Path
from easydict import EasyDict
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class imagenet_train_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(imagenet_train_dataset, self).__init__()
        self.cfg = EasyDict(cfg.copy())
        self.train_image_dir = Path(self.cfg.DATASET.TRAIN_DATA_PATH)
        self.token2id = {}
        self.train_dataset = self.load_data()

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        img_path, label = self.train_dataset[idx]

        return img_path, label

    def load_data(self):
        data_list = []
        img_folder_path = self.train_image_dir.glob('*')

        for img_folder in sorted(img_folder_path):
            token = img_folder.name
            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)
            label = self.token2id[token]
            for img_path in img_folder.glob('*.JPEG'):
                data_list.append([img_path, label])
        return data_list

    def get_token(self):
        return self.token2id

class imagenet_valid_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, token2id):
        self.cfg = EasyDict(cfg.copy())
        self.token2id = token2id
        self.valid_image_dir = Path(self.cfg.DATASET.VALID_DATA_PATH)
        self.valid_annotation_dir = Path(self.cfg.DATASET.VALID_ANNOTATION_PATH)

        self.valid_dataset = self.load_data(self.valid_image_dir, self.valid_annotation_dir)

    def __len__(self):
        return len(self.valid_dataset)

    def __getitem__(self, idx):
        img_path, label = self.valid_dataset[idx]

        return img_path, label

    def load_data(self, valid_image_dir, valid_annotation_dir):
        data_list = []
        img_list = valid_image_dir.glob('*.JPEG')

        for img_path in img_list:
            xml_path  = valid_annotation_dir / (img_path.stem + '.xml')
            tree = ET.parse(xml_path)
            token = tree.getroot().find('object').find('name').text
            label = self.token2id[token]
            data_list.append([img_path, label])

        return data_list

class custom_collate(object):
    def __init__(self, cfg=None, is_train=True):
        if cfg is None:
            assert "Please Input setting information"
        self.cfg = EasyDict(cfg.copy())
        transforms = []
        if is_train:
            for key, value in self.cfg.AUGMENT.TRAIN_TRANSFORMS.items():
                if 'OneOf' in key:
                    sub_transforms = [getattr(A,name)(**value[name]) for name in value if name != 'p']
                    p_value = value['p']
                    transforms.append(A.OneOf(sub_transforms, p=p_value))
                else:
                    if value is None:
                        transform = getattr(A, key)()
                    else:
                        transform = getattr(A, key)(**value)
                    transforms.append(transform)
            self.image_transform = A.Compose(transforms + [ToTensorV2()])
        else:
            for key, value in self.cfg.AUGMENT.VALID_TRANSFORMS.items():
                if value is None:
                    transform = getattr(A, key)()
                else:
                    transform = getattr(A, key)(**value)
                transforms.append(transform)
            self.image_transform = A.Compose(transforms + [ToTensorV2()])

    def __call__(self, batch):
        batch_images, batch_labels = [], []

        for image_path, label in batch:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.image_transform(image=image)["image"]

            batch_images.append(image)
            batch_labels.append(label)

        return torch.cat(batch_images).view(-1,3,self.cfg.DATA.IMAGE_SIZE,self.cfg.DATA.IMAGE_SIZE), torch.as_tensor(batch_labels)
