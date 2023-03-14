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
    csv_file = Path(r'COST-Challenge\tree_metadata_training_publish_BM.csv')
    file_list = []
    species_list = []

    random_subset = []
    complement_subset = []

    with open(csv_file, 'r') as f:
        f.readline()  # get rid of header
        for lix, line in enumerate(f.readlines()):
            elems = line.strip().split(',')
            file_list.append(str(csv_file.parent / ('train/' + elems[6])))
            species_list.append(elems[1])  # species
            # species_list.append(elems[2])  # genus
            if elems[8] == 'train':  # train/val split
                random_subset.append(lix)
            else:
                complement_subset.append(lix)


    target_classes = sorted(list(set(species_list)))
    print(f"Found {len(target_classes)} target classes.")
    species_id_list = [target_classes.index(i) for i in species_list]
    for spi, spn in enumerate(target_classes):
        print("\tClass %02d: %s" % (spi, spn))

    file_list = np.array(file_list)
    species_id_list = np.array(species_id_list)


    n_points = 16_384

    train_dataset = TreePointCloudsInFiles(file_list[random_subset], species_id_list[random_subset],
                                           max_points=n_points, use_columns=[], rot_aug=6, sampling='random')
    test_dataset = TreePointCloudsInFiles(file_list[complement_subset], species_id_list[complement_subset],
                                          max_points=n_points, use_columns=[], sampling='random')


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             num_workers=5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    model = PN2_Classification(num_features=0, num_target_classes=len(target_classes)).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_proportions = [100 * np.count_nonzero(species_id_list[random_subset] == curr_spec) / len(random_subset) for curr_spec in range(len(target_classes))]
    train_weights = torch.Tensor(1 / np.sqrt(np.array(train_proportions))).to(device)
    # train_weights = torch.Tensor([1] * len(target_classes)).to(device)

    def net_train():
        model.train()
        losses = []
        for i, data in enumerate(tqdm(train_loader, desc='Training in progress')):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            # loss = F.cross_entropy(out, data.y, weight=train_weights)
            loss = F.nll_loss(out, data.y, weight=train_weights)
            # oa = accuracy_score(np.argmax(out.detach().cpu(), axis=-1), data.y.cpu())
            loss.backward()
            losses.append(loss.detach().cpu().numpy().item())
            optimizer.step()
            # if (i + 1) % 1 == 0:
            #     print(f'[{i + 1}/{len(train_loader)}] Loss: {loss.to("cpu"):.4f}, OA: {oa:.4f}')
        return np.mean(losses)

    @torch.no_grad()
    def net_test(loader, ep_id):
        model.eval()
        true_labels = []
        predicted_labels = []
        for idx, data in enumerate(tqdm(loader, desc='Test in progress....')):
            data = data.to(device)
            outs = model(data)
            loss = F.nll_loss(outs, data.y, weight=train_weights).detach().cpu().numpy().item()
            pred, true = np.argmax(outs.detach().cpu(), axis=-1), data.y.cpu()
            true_labels.extend(true)
            predicted_labels.extend(pred)
        oa = accuracy_score(true_labels, predicted_labels)
        precs, recalls, f1s, support = precision_recall_fscore_support(true_labels, predicted_labels, zero_division=0)
        cm = confusion_matrix(true_labels, predicted_labels)
        return {'oa': oa, 'precs': precs, 'recalls': recalls, 'f1s': f1s, 'cm': cm, 'loss': loss}

    test_proportions = np.array([100 * np.count_nonzero(species_id_list[complement_subset] == curr_spec) / len(complement_subset) for curr_spec in range(len(target_classes))])
    print(f'\nPrior : {"|".join([f"{res:6.3f}" for res in test_proportions])}')

    basename = '3SAs_16kpts_lr1e-4_aug6_random'
    outdir = Path(rf'COST-Challenge\models\{basename}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cm_path = outdir / 'cm_latest.txt'
    meta_path = outdir / 'info.pickle'
    best_model_path = outdir / 'best.model'
    last_model_path = outdir / 'ep_{epoch:03d}.model'

    shutil.copy2(pn2_classifier.__file__, outdir)
    best_meanf1 = 0

    if os.path.exists(meta_path):
        meta = pickle.load(open(str(meta_path), 'rb'))
        last_epoch = max(meta.keys())
        model_path = meta[last_epoch]['path']
        print(f"Starting training from checkpoint at {model_path}.")
        model.load_state_dict(torch.load(model_path))
        best_meanf1 = max([meta[ep]['mean_f1'] for ep in meta.keys()])
    else:
        meta = OrderedDict()
        last_epoch = -1
        print("Starting training from scratch.")

    for epoch in range(last_epoch+1, 1001):
        meanloss = net_train()
        results = net_test(test_loader, epoch)

        torch.save(model.state_dict(), str(last_model_path).format(epoch=epoch))
        sys.stdout.flush()
        print(f'Epoch: {epoch:02d}, Test OA: {results["oa"]*100:.2f}%, Mean Loss: {meanloss:.4f}'
              f'\nPrior : {"|".join([f"{res:6.3f}" for res in test_proportions])} '
              f'|| Mean (weighted):'
              f'\nPrecis: {"|".join([f"{res:.4f}" if res > 0 else " "*6 for res in results["precs"]])} '
              f'|| {np.average(results["precs"], weights=test_proportions):.4f}'
              f'\nRecall: {"|".join([f"{res:.4f}" if res > 0 else " "*6 for res in results["recalls"]])} '
              f'|| {np.average(results["recalls"], weights=test_proportions):.4f}'
              f'\nF1s   : {"|".join([f"{res:.4f}" if res > 0 else " "*6 for res in results["f1s"]])} '
              f'|| {np.average(results["f1s"], weights=test_proportions):.4f}')

        if np.average(results["f1s"], weights=test_proportions) > best_meanf1:
            best_meanf1 = np.average(results["f1s"], weights=test_proportions)
            print("New best F1 score - updating best model file!")
            shutil.copy2(str(last_model_path).format(epoch=epoch), str(best_model_path))

        results = results | {
            'mean_f1': np.average(results["f1s"], weights=test_proportions),
            'mean_trainloss': meanloss,
            'path': str(last_model_path).format(epoch=epoch),
        }

        meta[epoch] = results
        with open(str(meta_path), 'wb') as of:
            pickle.dump(meta, of)
        with open(str(cm_path), 'w') as f:
            f.writelines('\n'.join('\t'.join('%d' % x for x in y) for y in results["cm"]))
        sys.stdout.flush()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot([ep['mean_trainloss'] for ep in meta.values()], 'orange', label='Training loss')
        ax.plot([ep['loss'] for ep in meta.values()], 'blue', label='Test loss')
        plt.show()

        cm_fig = plot_cm.main(results["cm"])
        cm_fig.show()
