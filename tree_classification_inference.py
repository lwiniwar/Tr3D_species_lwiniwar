import io
import os
import pickle
import random
import sys
import shutil
import tempfile
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

import pn2_classifier
from pn2_classifier import PN2_Classification
from pointcloud_dataset import read_las

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import fps

import plot_cm

class TreePointCloudsInFiles(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(self, point_file_list, label_list, max_points=200_000, use_columns=None, rot_aug=1, sampling='random'):
        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            csv_file (string): Path to CSV file with the labels
            use_columns (list[string]): Column names to add as additional input
            rot_aug (int): number of augmentations by rotation about the z-axis
            sampling (string): either "random" or "furthest"
        """
        self.files = point_file_list
        self.species = label_list
        self.max_points = max_points
        if use_columns is None:
            use_columns = []
        self.use_columns = use_columns
        self.rot_aug = rot_aug
        self.r = [
            np.array([[np.cos(rot_angle), -1 * np.sin(rot_angle), 0],
                      [np.sin(rot_angle), np.cos(rot_angle), 0],
                      [0, 0, 1]])
                     for rot_angle in np.linspace(0, 2*np.pi, rot_aug, endpoint=False)
        ]
        self.sampling = sampling
        assert self.sampling in ['random', 'furthest', 'height_bins']

        super().__init__()

    def __len__(self):
        return len(self.files) * self.rot_aug

    def __getitem__(self, idx):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        if torch.is_tensor(idx):
            idx = idx.tolist()

        rot_idx = idx // len(self.files)  # id of rotations (order is: sample1_0deg, sample1_180deg, sample2_0deg, ...)
        idx = idx - len(self.files) * rot_idx  # get the idx of the base file

        filename = str(self.files[idx])

        coords, attrs = read_las(filename, get_attributes=True)
        if coords.shape[0] >= self.max_points:
            if self.sampling == "random":
                use_idx = np.random.choice(coords.shape[0], self.max_points, replace=False)
            elif self.sampling == "furthest":  # very slow, not tested
                use_idx = fps(torch.from_numpy(coords).to(device), ratio=(self.max_points / coords.shape[0])).cpu().numpy()
                assert use_idx.shape[0] == self.max_points
            elif self.sampling == "height_bins":
                coords = np.vstack((np.array([0, 0, 0]), coords))  # add zero point to fill up later
                min_z = np.min(coords[1:, 2])
                range_z = np.ptp(coords[1:, 2])
                use_idx = []
                num_bins = 16
                points_per_bin = self.max_points // num_bins
                assert self.max_points % num_bins == 0, "number of points has to be divisible by number of bins"
                for bin in range(num_bins):
                    lower_z = min_z + bin * range_z / num_bins
                    upper_z = lower_z + range_z / num_bins
                    idx_in_slice = np.nonzero(np.logical_and(coords[:, 2] >= lower_z, coords[:, 2] < upper_z))[0]
                    if idx_in_slice.shape[0] == 0:
                        # no points in this bin
                        use_idx.append([0] * points_per_bin)
                    elif idx_in_slice.shape[0] < points_per_bin:
                        use_idx.append(idx_in_slice) # each point once
                        use_idx.append(np.random.choice(idx_in_slice,
                                                        points_per_bin - idx_in_slice.shape[0],
                                                        replace=True))  #fill up with random points
                    else:
                        use_idx.append(np.random.choice(idx_in_slice, points_per_bin, replace=False))
                # use_idx_bk = use_idx.copy()
                use_idx = np.concatenate(use_idx)
        else:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=True)
        if len(self.use_columns) > 0:
            x = np.empty((self.max_points, len(self.use_columns)), np.float32)
            for eix, entry in enumerate(self.use_columns):
                x[:, eix] = attrs[entry][use_idx]
        else:
            x = None
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        if rot_idx >= 1:
            coords = np.dot(self.r[rot_idx], coords.T).T

        # print(use_idx.dtype)
        sample = Data(x=torch.from_numpy(x).float() if x is not None else None,
                      # y=torch.from_numpy(self.species[idx]).int(),
                      y=torch.Tensor([self.species[idx]]).type(torch.LongTensor),
                      pos=torch.from_numpy(coords[use_idx, :]).float())

        return sample

if __name__ == '__main__':


    file_list = list(Path(r'COST-Challenge\test').glob("*.las"))
    tree_id_list = [int(item.name.split('.las')[0]) for item in file_list]

    csv_file = Path(r'COST-Challenge\tree_metadata_training_publish.csv')
    species_list = []

    with open(csv_file, 'r') as f:
        f.readline()  # get rid of header
        for lix, line in enumerate(f.readlines()):
            elems = line.strip().split(',')
            species_list.append(elems[1])  # species
    class_list = sorted(list(set(species_list)))

    # n_points = 4_096
    n_points = 16_384
    # n_points = 8_192

    test_dataset = TreePointCloudsInFiles(file_list, tree_id_list,
                                          max_points=n_points, use_columns=[], sampling='random')

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             num_workers=5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    model = PN2_Classification(num_features=0, num_target_classes=33).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    @torch.no_grad()
    def net_test(loader):
        model.eval()
        tree_ids = []
        predicted_labels = []
        for idx, data in enumerate(tqdm(loader, desc='Inference in progress....')):
            data = data.to(device)
            outs = model(data)
            pred = np.argmax(outs.detach().cpu(), axis=-1)
            tree_ids.extend(data.y.detach().cpu().numpy())
            predicted_labels.extend(pred)
        return tree_ids, predicted_labels

    basename = '3SAs_8kpts_lr1e-4_aug6_random'
    outdir = Path(rf'COST-Challenge\models\{basename}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    test_result_path = outdir / 'test.csv'
    best_model_path = outdir / 'best.model'

    print(f"Loading checkpoint at {best_model_path}.")
    model.load_state_dict(torch.load(best_model_path))

    results = net_test(test_loader)
    with open(test_result_path, 'w') as f:
        f.write(f"treeID,predicted_species\n")
        for tree_id, class_pred in zip(results[0], results[1]):
            name_pred = class_list[class_pred]
            f.write(f"{tree_id},{name_pred}\n")
