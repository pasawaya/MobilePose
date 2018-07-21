
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import json
from pprint import pprint


class MPII(Dataset):
    def __init__(self, root, input_size=256, train=True):
        self.root = root
        self.train = train
        self.input_size = input_size
        self.flag = int(train)

        annotations_name = 'mpii_train.json' if train else 'mpii_valid.json'
        self.annotations_path = os.path.join(self.root, annotations_name)
        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[str(idx)]
        pprint(annotation)
        return annotation

    def generate_annotations(self):
        n_joints = 16

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
                            if 'is_visible' in str(points.dtype):
                                ids = [str(p_id[0, 0]) for p_id in points['id'][0]]
                                x = [int(p_x[0, 0]) for p_x in points['x'][0]]
                                y = [int(p_y[0, 0]) for p_y in points['y'][0]]
                                vis = [1 if p_vis else 0 for p_vis in points['is_visible'][0]]
                                joints = {p_id: (p_x, p_y, p_vis) for p_id, p_x, p_y, p_vis in zip(ids, x, y, vis)}

                                for p in range(n_joints):
                                    if str(p) not in joints:
                                        joints[str(p)] = (-1, -1, 0)

                                data[i] = {'image_path': image_path,
                                           'joints': joints,
                                           'flag': self.flag}
                                i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)


mpii = MPII('../data/MPII')
data = mpii[0]
