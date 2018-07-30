
from torch.utils.data import Dataset

import os
import json

from utils.dataset_utils import *
from skimage.io import imread
from scipy.io import loadmat


train_ratio = 0.95


class MPII(Dataset):
    def __init__(self, root='data/MPII', transformer=None, output_size=256, train=True, subset_size=None,
                 sigma=7, stride=4, offset=0, include_background=True):
        self.root = root
        self.train = train
        self.output_size = output_size
        self.flag = int(train)
        self.transformer = transformer
        self.n_joints = 14
        self.sigma = sigma
        self.stride = stride
        self.offset = offset
        self.background = include_background

        self.annotations_path = os.path.join(self.root, 'mpii.json')

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

        n = len(self.annotations)
        self.start_idx = 0 if train else int(np.floor(train_ratio * n))
        self.size = int(np.floor(train_ratio * n)) if train else n - self.start_idx

        if subset_size is not None:
            self.size = subset_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = self.start_idx + idx
        path = self.annotations[str(idx)]['image_path']
        labels = self.annotations[str(idx)]['joints']

        image = imread(path).astype(np.float32)
        x, y, visibility = to_numpy(labels)

        if self.transformer is not None:
            image, x, y, visibility = self.transformer(image, x, y, visibility)

        label_map = compute_label_map(x, y, visibility, self.output_size, self.sigma, self.stride, self.offset, self.background)
        center_map = compute_center_map(size=self.output_size)
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        meta = torch.from_numpy(np.squeeze(np.hstack([x, y]))).float()
        return image, label_map, center_map, meta

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
