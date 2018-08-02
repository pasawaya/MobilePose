
from torch.utils.data import Dataset

import os
import json

from utils.dataset_utils import *
from skimage.io import imread
from scipy.io import loadmat


train_ratio = 0.95


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
        self.transformer = transformer
        self.n_joints = 14
        self.sigma_center = sigma_center
        self.sigma_label = sigma_label
        self.label_size = label_size

        self.annotations_path = os.path.join(self.root, 'mpii.json')

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

        n = len(self.annotations)
        self.start_idx = 0 if train else int(np.floor(train_ratio * n))
        self.size = int(np.floor(train_ratio * n)) if train else n - self.start_idx

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = self.start_idx + idx
        path = self.annotations[str(idx)]['image_path']
        labels = self.annotations[str(idx)]['joints']

        image = imread(path).astype(np.float32)
        x, y, visibility = self.dict_to_numpy(labels)

        if self.transformer is not None:
            image, x, y, visibility, unnormalized = self.transformer(image, x, y, visibility)

        label_map = compute_label_map(x, y, self.output_size, self.label_size, self.sigma_label)
        center_map = compute_center_map(x, y, self.output_size, self.sigma_center)
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        meta = torch.from_numpy(np.squeeze(np.hstack([x, y]))).float()

        image = torch.unsqueeze(image, 0).repeat(self.T, 1, 1, 1)
        unnormalized = torch.unsqueeze(unnormalized, 0).repeat(self.T, 1, 1, 1)
        label_map = label_map.repeat(self.T, 1, 1, 1)
        return image, label_map, center_map, meta, unnormalized

    def generate_annotations(self):
        mpii_joints = 16

        contents = loadmat(os.path.join(self.root, 'annotations.mat'))['RELEASE']
        data = {}
        i = 0

        for annotation, flag in zip(contents['annolist'][0, 0][0], contents['img_train'][0, 0][0]):
            image_name = annotation['image']['name'][0, 0][0]
            image_path = os.path.join(self.root, 'images', image_name)
            annorect = annotation['annorect']
            if 'annopoints' in str(annorect.dtype):
                annopoints = annorect['annopoints'][0]
                for annopoint in annopoints:
                    if len(annopoint) > 0:
                        points = annopoint['point'][0, 0]
                        ids = [str(p_id[0, 0]) for p_id in points['id'][0]]
                        if 'is_visible' in str(points.dtype) and len(ids) == mpii_joints:
                            x = [int(p_x[0, 0]) for p_x in points['x'][0]]
                            y = [int(p_y[0, 0]) for p_y in points['y'][0]]
                            vis = [1 if p_vis else 0 for p_vis in points['is_visible'][0]]

                            ignored = ['6', '7']   # Ignore pelvis and thorax
                            shifted = ['14', '15']   # Indices to replace pelvis and thorax
                            visible = ['8', '9']   # Head and neck indices to always set visible

                            joints = {}
                            for p_id, p_x, p_y, p_vis in zip(ids, x, y, vis):
                                if p_id in visible:
                                    joints[p_id] = (p_x, p_y, 1)
                                elif p_id in shifted:
                                    joints[str(int(p_id) - 8)] = (p_x, p_y, p_vis)
                                elif p_id not in ignored:
                                    joints[p_id] = (p_x, p_y, p_vis)

                            data[i] = {'image_path': image_path,
                                       'joints': joints}
                            i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)

    @staticmethod
    def dict_to_numpy(data):
        n = len(data)
        x, y, vis = np.zeros(n), np.zeros(n), np.zeros(n)
        for p in range(n):
            x[p] = data[str(p)][0]
            y[p] = data[str(p)][1]
            vis[p] = data[str(p)][2]
        return x, y, vis
