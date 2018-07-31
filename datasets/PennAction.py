
from torch.utils.data import Dataset

import os
import json
import glob

from utils.dataset_utils import *
from skimage.io import imread
from scipy.io import loadmat


class PennAction(Dataset):
    def __init__(self, T, root='../data/PennAction', transformer=None, output_size=256, train=True, subset_size=None,
                 sigma=7, stride=4, offset=0, include_background=True):
        self.T = T
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

        self.annotations_path = os.path.join(self.root, 'penn.json')

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        frames = self.load_video(idx)
        x, y, visibility, bbox = self.load_annotation(idx)

        if self.transformer is not None:
            frames, x, y, visibility, bbox = self.transformer(frames, x, y, visibility, bbox)

        label_map = compute_label_map(x, y, self.output_size, self.sigma, self.stride, self.offset, self.background)
        center_map = compute_center_map(self.output_size)
        meta = torch.from_numpy(np.squeeze(np.hstack([x, y]))).float()
        return frames, label_map, center_map, meta

    def load_annotation(self, idx):
        annotations_path = self.annotations[str(idx)]['annotations_path']
        start = int(self.annotations[str(idx)]['start_index'])

        annotations = loadmat(annotations_path)
        x = annotations['x'][start:start + self.T, :]
        y = annotations['y'][start:start + self.T, :]
        vis = annotations['visibility'][start:start + self.T, :]
        bbox = annotations['bbox'][start:start + self.T, :]

        x, y, vis = self.infer_neck_annotation(x, y, vis)
        x, y, vis = self.reorder_joints(x, y, vis)

        return x, y, vis, bbox

    def load_video(self, idx):
        frames_root = self.annotations[str(idx)]['frames_root']
        start = int(self.annotations[str(idx)]['start_index'])

        frame_paths = sorted(glob.glob(frames_root))[start:start + self.T]
        frames = [imread(frame).astype(np.float32) for frame in frame_paths]
        return frames

    def generate_annotations(self):
        data = {}
        i = 0

        annotations_directory = os.path.join(self.root, 'labels')
        for file in os.listdir(annotations_directory):
            filename = os.fsdecode(file)
            if filename.endswith('.mat'):
                annotation_path = os.path.join(annotations_directory, filename)
                annotations = loadmat(annotation_path)
                video_id = filename.split('.')[0]
                frames_root = os.path.join(self.root, 'frames', video_id, '*')
                _, _, n = annotations['dimensions'][0]
                indices = np.arange(0, n, self.T)[:-1]   # Exclude last range, may not be `T` long
                for start_index in indices:
                    data[i] = {'annotations_path': annotation_path,
                               'frames_root': frames_root,
                               'start_index': str(start_index)}
                    i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)

    @staticmethod
    def infer_neck_annotation(x, y, vis):
        neck_x = np.expand_dims(0.5 * x[:, 0] + 0.25 * (x[:, 1] + x[:, 2]), 1)
        neck_y = np.expand_dims(0.2 * y[:, 0] + 0.4 * (y[:, 1] + y[:, 2]), 1)
        neck_vis = np.expand_dims(np.floor((vis[:, 0] + vis[:, 1] + vis[:, 2]) / 3.), 1)

        x = np.hstack([x, neck_x])
        y = np.hstack([y, neck_y])
        vis = np.hstack([vis, neck_vis])
        return x, y, vis

    @staticmethod
    def reorder_joints(x, y, vis):
        mpii_order = [12, 10, 8, 7, 9, 11, 3, 5, 13, 0, 6, 4, 2, 1]
        return x[:, mpii_order], y[:, mpii_order], vis[:, mpii_order]
