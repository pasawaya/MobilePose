
from torch.utils.data import Dataset

import os
import json

from utils.dataset_utils import *
from utils.augmentation import *
from skimage.io import imread
from scipy.io import loadmat


class MPII(Dataset):
    def __init__(self, T,
                 root='data/MPII',
                 transformer=None,
                 train=True,
                 output_size=256,
                 sigma_center=21,
                 sigma_label=2,
                 label_size=31):

        self.T = T
        self.root = root
        self.train = train
        self.output_size = output_size
        self.flag = int(train)
        self.n_joints = 14
        self.sigma_center = sigma_center
        self.sigma_label = sigma_label
        self.label_size = label_size
        self.transformer = transformer

        annotations_label = 'train_' if train else 'valid_'
        self.annotations_path = os.path.join(self.root, annotations_label + 'annotations.json')

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = self.annotations[str(idx)]['image_path']
        image = imread(path).astype(np.float32)
        x, y, visibility = self.load_annotation(idx)

        if self.transformer is not None:
            image, x, y, visibility, unnormalized = self.transformer(image, x, y, visibility)
        else:
            unnormalized = Transformer.to_torch(image)
            image = Transformer.to_torch(image)

        label_map = compute_label_map(x, y, self.output_size, self.label_size, self.sigma_label)
        center_map = compute_center_map(x, y, self.output_size, self.sigma_center)
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        meta = torch.from_numpy(np.squeeze(np.hstack([x, y]))).float()

        image = torch.unsqueeze(image, 0).repeat(self.T, 1, 1, 1)
        unnormalized = torch.unsqueeze(unnormalized, 0).repeat(self.T, 1, 1, 1)
        label_map = label_map.repeat(self.T, 1, 1, 1)
        return image, label_map, center_map, meta, unnormalized

    def load_annotation(self, idx):
        labels = self.annotations[str(idx)]['joints']
        x, y, visibility = self.dict_to_numpy(labels)
        return x, y, visibility

    def generate_annotations(self):
        data = {}
        i = 0

        annotations = loadmat(os.path.join(self.root, 'annotations.mat'))['RELEASE']

        for image_idx in range(annotations['img_train'][0][0][0].shape[0]):
            if self.train == self.is_train(annotations, image_idx):
                image_path = os.path.join(self.root, 'images', self.get_image_name(annotations, image_idx))
                for person_idx in range(self.n_people(annotations, image_idx)):
                    c, s = self.location(annotations, image_idx, person_idx)
                    if not c[0] == -1:
                        joints = self.get_person_joints(annotations, image_idx, person_idx)

                        if len(joints) > 0:
                            ignored = ['6', '7']  # Ignore pelvis and thorax
                            shifted = ['14', '15']  # Indices to replace pelvis and thorax

                            for idx_ignored, idx_shifted in zip(ignored, shifted):
                                joints[idx_ignored] = joints[idx_shifted]
                                del joints[idx_shifted]

                            data[i] = {'image_path': image_path,
                                       'joints': joints}
                            i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)

    @staticmethod
    def get_person_joints(annotations, image_idx, person_idx):
        mpii_joints = 16
        joints = {}
        image_info = annotations['annolist'][0][0][0]['annorect'][image_idx]
        if 'annopoints' in str(image_info.dtype) and image_info['annopoints'][0][person_idx].size > 0:
            person_info = image_info['annopoints'][0][person_idx][0][0][0][0]
            if len(person_info) == mpii_joints:
                for i in range(mpii_joints):
                    p_id, p_x, p_y = person_info[i]['id'][0][0], \
                                     int(person_info[i]['x'][0][0]),\
                                     int(person_info[i]['x'][0][0])
                    vis = 1
                    if 'is_visible' in person_info.dtype.fields:
                        vis = person_info[i]['is_visible']
                        vis = int(vis[0][0]) if len(vis) > 0 else 1

                    joints[str(p_id)] = (p_x, p_y, vis)
        return joints

    @staticmethod
    def get_image_name(annotations, image_idx):
        return str(annotations['annolist'][0][0][0]['image'][:][image_idx][0][0][0][0])

    @staticmethod
    def dict_to_numpy(data):
        n = len(data)
        x, y, vis = np.zeros(n), np.zeros(n), np.zeros(n)
        for p in range(n):
            x[p] = data[str(p)][0]
            y[p] = data[str(p)][1]
            vis[p] = data[str(p)][2]
        return x, y, vis

    # Functions below taken from https://github.com/umich-vl/pose-hg-train/blob/master/src/misc/mpii.py
    @staticmethod
    def n_people(annot, image_idx):
        example = annot['annolist'][0][0][0]['annorect'][image_idx]
        if len(example) > 0:
            return len(example[0])
        else:
            return 0

    @staticmethod
    def is_train(annotations, image_idx):
        return (annotations['img_train'][0][0][0][image_idx] and
                annotations['annolist'][0][0][0]['annorect'][image_idx].size > 0 and
                'annopoints' in annotations['annolist'][0][0][0]['annorect'][image_idx].dtype.fields)

    @staticmethod
    def location(annot, image_idx, person_idx):
        example = annot['annolist'][0][0][0]['annorect'][image_idx]
        if ((not example.dtype.fields is None) and
                'scale' in example.dtype.fields and
                example['scale'][0][person_idx].size > 0 and
                example['objpos'][0][person_idx].size > 0):
            scale = example['scale'][0][person_idx][0][0]
            x = example['objpos'][0][person_idx][0][0]['x'][0][0]
            y = example['objpos'][0][person_idx][0][0]['y'][0][0]
            return np.array([x, y]), scale
        else:
            return [-1, -1], -1
