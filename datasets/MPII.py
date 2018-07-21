
from torch.utils.data import Dataset

import os
import json

from utils.dataset_utils import *
from skimage.io import imread
from scipy.io import loadmat


class MPII(Dataset):
    def __init__(self, root='data/MPII', transformer=None, output_size=256, train=True):
        self.root = root
        self.train = train
        self.output_size = output_size
        self.flag = int(train)
        self.transformer = transformer
        self.n_joints = 14

        annotations_name = 'mpii_train.json' if train else 'mpii_valid.json'
        self.annotations_path = os.path.join(self.root, annotations_name)

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

        mean_name = 'means.npy'
        mean_path = os.path.join(self.root, mean_name)
        if not os.path.isfile(mean_path):
            mean, std = self.compute_mean()
            np.save(mean_path, np.array([mean, std]))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = self.annotations[str(idx)]['image_path']
        labels = self.annotations[str(idx)]['joints']

        image = imread(path).astype(np.float32)
        x, y, visibility = to_numpy(labels)

        if self.transformer is not None:
            image, x, y, visibility = self.transformer(image, x, y, visibility)

        label_map = compute_label_map(x, y, visibility, self.output_size)
        center_map = compute_center_map(self.output_size)
        return image, label_map, center_map, (x, y, visibility)

    def generate_annotations(self):
        mpii_joints = 16

        contents = loadmat(os.path.join(self.root, 'annotations.mat'))['RELEASE']
        data = {}
        i = 0

        for annotation, flag in zip(contents['annolist'][0, 0][0], contents['img_train'][0, 0][0]):
            if flag == self.flag:
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

                                for p in range(mpii_joints - len(ignored)):
                                    if str(p) not in joints:
                                        joints[str(p)] = (-1, -1, 0)

                                data[i] = {'image_path': image_path,
                                           'joints': joints}
                                i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)

    def compute_mean(self):
        mean, std = np.zeros(3, dtype=np.float), np.zeros(3, dtype=np.float)

        for i in range(len(self.annotations)):
            print(i)
            path = self.annotations[str(i)]['image_path']
            im = imread(path)
            im = np.moveaxis(im, 2, 0)
            im = im.reshape(im.shape[0], -1)
            mean += np.mean(im, axis=1)
            std += np.std(im, axis=1)
        mean = (mean / 255.) / float(len(self.annotations))
        std = (std / 255.) / float(len(self.annotations))
        return mean, std
